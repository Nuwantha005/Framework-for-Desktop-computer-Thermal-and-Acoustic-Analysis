# I/O Module State
**Last modified**: 2026-02-25

## Files
- `core/io/geometry_io.py` — `GeometryReader` (static: `read_json()`, `read_xy()`, `read()`); standalone `generate_rectangle()`, `generate_circle()`
- `core/io/case_loader.py` — `CaseLoader` (static: `load()`, `validate()`, `load_case()`) — parses YAML, creates Scene via GeometryFactory, added 3D specific loading paths `_build_scene_3d`.
- `core/io/case.py` — `Case` dataclass: unified wrapper holding Scene + SimulationConfig + case_dir; cached mesh property; convenience accessors (v_inf, aoa, x_range, y_range, resolution, etc.); `create_solver()`, `get_fluid_state()`
- `core/io/case_exporter.py` — `CaseExporter` (class methods: `from_scene()`, `from_mesh()`) — exports to JSON/YAML
- `core/config/schemas.py` — Pydantic models: `SimulationConfig`, `ComponentConfig`, `SolverConfig`, `FluidConfig`, `VisualizationConfig`, `GeometryConfig`, `TransformConfig`, `OutputConfig`, `BoundaryConditionType`

## Public API
- `CaseLoader.load_case(case_dir, mesh_level_index=0) -> Case`
- `Case.create_solver() -> Solver`
- `Case.get_fluid_state() -> FluidState`
- `Case.reload_at_level(index) -> Case`
- `GeometryReader.read(filepath) -> Mesh`

## Data Flow
`case.yaml` → `CaseLoader.load()` → parses YAML → validates with Pydantic schemas → creates Components via `GeometryFactory` → builds `Scene` → wraps in `Case`

## Dependencies
- Internal: `core.geometry` (Mesh, Component, Scene, Transform, GeometryFactory), `core.config.schemas`
- External: pydantic, pyyaml, pathlib, numpy

## What's Next
- 3D case support (volume mesh references)
- HDF5 result export via h5py

## Known Issues
- None observed
