# Post-processing Module State
**Last modified**: 2026-02-25

## Files
- `fields.py` — `ScalarField`, `VectorField` dataclasses; `FieldData` (grid-based field registry: `add_scalar()`, `add_vector()`, `get_scalar()`, `get_vector()`, `available`)
- `fluid.py` — `ReferenceType` enum; `ReferenceCondition` dataclass; `FluidState` dataclass with factories: `air_standard()`, `water_standard()`, `incompressible()`, `from_dict()`
- `pipeline.py` — `PostProcessor` ABC (`requires`, `produces`, `process()`); `ProcessorPipeline` (topological sort, `add()`, `run()`, `available_outputs()`)
- `pressure.py` — `PressureProcessor` (requires: velocity; produces: pressure, pressure_gauge, Cp, total_pressure); `PressureGradientProcessor`
- `velocity_potential.py` — `VelocityPotentialProcessor` (requires: velocity; produces: velocity_potential, stream_function); `VorticityProcessor` (produces: vorticity)
- `surface.py` — `SurfaceData` dataclass; `SurfaceDataExtractor` (`extract()`, `extract_by_component()`, `interpolate_to_arc_length()`)

## Public API
- `ProcessorPipeline().add(PressureProcessor()).add(VorticityProcessor()).run(fields, fluid)`
- `FieldData(XX, YY)` → `.add_scalar("name", data)` → `.pressure`, `.velocity`, etc.
- `SurfaceDataExtractor(mesh, solver).extract() -> SurfaceData`

## Data Flow
Solver → `VelocityField2D.compute()` → `FieldData(XX, YY)` + velocity vectors → `ProcessorPipeline.run()` → adds derived fields (pressure, Cp, φ, ψ, ω) → passed to Visualizer

## Dependencies
- Internal: `core.geometry.mesh.Mesh`
- External: numpy, scipy (for stream function integration)

## What's Next
- Heat transfer coefficient processor (thermal BL coupling)
- Force integration processor (lift/drag from surface pressure)

## Known Issues
- None observed
