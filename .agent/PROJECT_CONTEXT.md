# Panel Method Solver — Project Context
**Last updated**: 2026-03-01
**Updated by**: agent — Linear source/doublet solver implementation

## Current Focus
> **Phase: Viscous BL solver (Von Kármán integral)**
>
> All five 2D panel solvers now implemented (constant source, linear source,
> linear vortex, Dirichlet doublet/Morino, linear source/doublet). Linear vortex
> and linear source provide best accuracy (~3.7–3.8% Vt error vs OpenFOAM).
> Constant-order Dirichlet methods (constant doublet: 50.87%, linear source/doublet: 50.83%)
> match each other as expected since dμ/ds extraction is constant-order for velocity.
> Von Kármán momentum integral BL solver now implemented with multiple velocity
> profiles (Blasius, Thwaites, Pohlhausen, Falkner-Skan, Power Law) and transition
> criteria (Michel, e^N). Next: thermal BL (BDIM from Gao 2013), viscous-inviscid
> coupling, and 3D panel methods with PyVista visualization.

## What This Project Is
A 2D panel method solver for potential flow analysis, part of a Final Year Project to develop a framework for desktop computer thermal and acoustic analysis. The solver uses constant-strength source panels with Neumann boundary conditions (no-penetration). An OpenFOAM-based validation pipeline compares panel method results against CFD. Future phases will add vortex panels, boundary layer solvers, and thermal coupling.

## Current State
- [x] Constant-strength source panel solver (Katz & Plotkin formulation)
- [x] Linear-strength source panel solver (continuous node-based formulation)
- [x] Linear-strength vortex panel solver (zero-circulation closure, direct Vt)
- [x] Dirichlet doublet panel solver (Morino source+doublet, internal-potential BC)
- [x] Linear source/doublet panel solver (Morino, linear-strength, K&P §11.5.1)
- [x] Case file I/O (YAML cases + JSON geometries)
- [x] Geometry module (parametric circle/rectangle/rounded_rectangle, STL export)
- [x] OpenFOAM validation pipeline (meshing, grid independence, comparison)
- [x] Visualization (contours, streamlines, Cp, surface envelopes, comparison)
- [x] Post-processing pipeline (pressure, velocity potential, vorticity, stream function)
- [ ] Quadratic strength panels
- [ ] Viscous boundary layer solver (Von Kármán integral)
- [ ] Thermal boundary layer solver (BDIM)
- [ ] Coupled inviscid-viscous iteration
- [ ] 3D panel method
- [ ] Actuator disk model (fans)

## Module Map
```
case.yaml ──► CaseLoader ──► Case ──► Scene ──► Mesh (assemble)
                                │                    │
                                ▼                    ▼
                          create_solver()    SourcePanelSolver / LinearSourcePanelSolver
                                             / LinearVortexPanelSolver / DirichletDoubletSolver
                                             / LinearSourceDoubletSolver
                                                     │
                                              solve() │
                                                     ▼
                              VelocityField2D ◄── velocity_at(points)
                                     │
                                     ▼
                              ProcessorPipeline
                              (pressure, Cp, φ, ψ, ω)
                                     │
                                     ▼
                        Visualizer / ComparisonVisualizer
                                     │
                                     ▼
                              out/*.png, out/*.csv
```

## Key Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| numpy | Array operations, linear algebra | >=1.24.0 |
| scipy | Sparse solvers, interpolation | >=1.11.0 |
| pydantic | Config schema validation | >=2.0.0 |
| pyyaml | YAML case file parsing | >=6.0 |
| shapely | 2D geometry operations | >=2.0.0 |
| gmsh | Meshing backend | >=4.11.0 |
| pygmsh | Python interface to gmsh | >=7.1.0 |
| meshio | Mesh file I/O | >=5.3.0 |
| matplotlib | 2D plotting | >=3.7.0 |
| pyvista | 3D visualization (VTK) | >=0.42.0 |
| pytest | Testing framework | >=7.4.0 |
| hypothesis | Property-based testing | >=6.82.0 |
| ruff | Linting | >=0.1.0 |
| mypy | Type checking | >=1.5.0 |
| rich | Console formatting | >=13.5.0 |
| joblib | Parallel execution | >=1.3.0 |
| h5py | HDF5 result storage | >=3.9.0 |
| foamlib | OpenFOAM case generation (implicit) | — |
| trimesh | STL geometry export (implicit) | — |

## Source Structure
```
src/
├── __init__.py                          # __version__ = "0.1.0"
├── core/
│   ├── config/schemas.py                # Pydantic models (SimulationConfig, etc.)
│   ├── geometry/
│   │   ├── primitives.py                # Point3D, Vector3D, rotation helpers
│   │   ├── mesh.py                      # Mesh dataclass (nodes, panels, geometry)
│   │   ├── component.py                 # Transform, Component
│   │   ├── scene.py                     # Scene (multi-component assembly)
│   │   ├── generators.py                # generate_circle/rectangle/rounded_rectangle
│   │   └── factory.py                   # GeometryFactory (registry → generators)
│   └── io/
│       ├── geometry_io.py               # GeometryReader (JSON/XY formats)
│       ├── case_loader.py               # CaseLoader (YAML → Scene + Config)
│       ├── case.py                      # Case dataclass (unified wrapper)
│       └── case_exporter.py             # CaseExporter (mesh → JSON/YAML)
├── solvers/
│   ├── base.py                          # Solver ABC
│   ├── factory.py                       # SolverFactory (registry)
│   └── panel2d/
│       ├── base.py                      # PanelSolver2D ABC, PanelMethodConfig
│       ├── spm.py                       # SourcePanelSolver (constant source)
│       ├── linear_source_solver.py       # LinearSourcePanelSolver
│       ├── linear_vortex_solver.py       # LinearVortexPanelSolver
│       ├── dirichlet_doublet_solver.py   # DirichletDoubletSolver (Morino)
│       ├── linear_source_doublet_solver.py # LinearSourceDoubletSolver (linear Morino)
│       └── influences/
│           ├── source.py                # Constant source influence coefficients
│           ├── linear_source.py         # Linear source influence coefficients
│           ├── linear_vortex.py         # Linear vortex influence coefficients
│           ├── doublet.py               # Constant doublet potential & velocity influences
│           └── linear_doublet.py        # Linear doublet + source potential & velocity influences
├── postprocessing/
│   ├── fields.py                        # FieldData, ScalarField, VectorField
│   ├── fluid.py                         # FluidState, ReferenceCondition
│   ├── pipeline.py                      # PostProcessor ABC, ProcessorPipeline
│   ├── pressure.py                      # PressureProcessor, PressureGradientProcessor
│   ├── velocity_potential.py            # VelocityPotentialProcessor, VorticityProcessor
│   └── surface.py                       # SurfaceData, SurfaceDataExtractor
├── visualization/
│   ├── visualizer.py                    # Visualizer facade, OutputManager
│   ├── field2d.py                       # VelocityField2D (caching, body masking)
│   ├── comparison.py                    # ComparisonVisualizer, metrics
│   ├── panel2d.py                       # PanelVisualizer2D (legacy)
│   ├── mesh_plot.py                     # MeshPlotter, quick_plot_* helpers
│   ├── surface_envelope.py              # Surface envelope plots
│   ├── streamlines.py                   # StreamlineVisualizer (multiprocessing)
│   └── plotters/{contours,streamlines}.py
└── test/
    ├── test_geometry_foundation.py      # pytest: geometry, mesh, scene, transforms
    └── test_foundation.py               # Manual sanity checks (not pytest)
```

## Architecture Decisions
- **Dataclass containers**: Mesh, Component, Scene, Case, FieldData are all dataclasses
- **ABC solver hierarchy**: Solver → PanelSolver2D → SourcePanelSolver (template method)
- **Factory registries**: GeometryFactory and SolverFactory use string-keyed registries
- **Array convention**: All coordinate arrays are (N, 3) with z=0 for 2D; float64 nodes, int32 panels
- **Case format**: YAML for simulation config, JSON for raw geometry; parametric shapes preferred
- **Pipeline pattern**: ProcessorPipeline uses topological sort for dependency resolution
- **No packaging (was)**: Scripts used sys.path.insert; now pyproject.toml added for editable install

## Known Issues / Tech Debt
- Root README.md is outdated (reflects Phase 1 state)
- `test_foundation.py` uses print-based assertions, not pytest
- `spm.py` has debug `print()` statements in `compute_source_influence_matrices`
- No conftest.py or pytest fixtures
- Placeholder directories exist for panel3d, thermal, actuator (boundary_layer now populated)
- `foamlib` and `trimesh` not listed in requirements.txt but imported in validation/

## Agent Infrastructure
- **`.github/copilot-instructions.md`** — Copilot custom instructions (auto-loaded)
- **`.agent/prompts/`** — 5 reusable task prompts:
  - `implement-solver.md` — new solver implementation (6-step pattern)
  - `validate-vt.md` — tangential velocity debugging loop
  - `implement-bl-solver.md` — Von Kármán BL solver
  - `implement-thermal-bl.md` — BDIM thermal BL solver
  - `port-to-3d.md` — 3D extension with PyVista
- **`.agent/decisions/`** — architectural decision records (with template)
- **MCP servers**: Python (fyp env), Pylance (auto-imports, docstrings, refactoring)
- **Reference materials**: K&P Ch 9–12 (markdown), code snippets (MATLAB+Python),
  Von Kármán notes, Gao 2013 paper (convert to markdown before use)

## Roadmap
1. **Viscous BL solver** — Von Kármán integral, multiple velocity profiles
2. **Thermal BL solver** — Reynolds analogy baseline, then full BDIM
3. **3D panel methods** — Mesh3D, source3d, PyVista visualization
5. **3D BL + thermal** — extend BL/thermal solvers for 3D surfaces
6. **Application** — heat transfer computation for computer components

## Environment
- Package manager: mamba (miniforge3)
- Environment name: fyp
- Python version: 3.13.11
- OS: Arch Linux
- Installed as editable: `pip install -e .` (pyproject.toml)
