# Architecture Decisions

## ADR-001: (N, 3) Array Convention for 2D
**Date**: Project inception
**Decision**: All coordinate arrays use shape `(N, 3)` with `z=0` for 2D problems — nodes, centers, normals, tangents, freestream.
**Reason**: Avoids separate 2D/3D code paths. When 3D panels are implemented, the same data structures and interfaces work without modification. Minor memory overhead is negligible.

## ADR-002: Dataclass Containers Over Active Objects
**Date**: Project inception
**Decision**: Core data structures (`Mesh`, `Component`, `Scene`, `Case`, `FieldData`) are implemented as Python `@dataclass` rather than classes with complex methods.
**Reason**: Keeps data and behavior separate. Data structures are transparent and easy to inspect. Processing logic lives in dedicated functions and classes (solvers, processors, visualizers).

## ADR-003: Solver ABC with Template Method
**Date**: Phase 6 refactoring
**Decision**: `Solver` → `PanelSolver2D` → `SourcePanelSolver` hierarchy using abstract base classes. `PanelSolver2D.solve()` implements the template method pattern calling four abstract steps.
**Reason**: Adding new panel methods (vortex, linear strength) requires implementing only the four step methods. The solve workflow, property accessors, and validation infrastructure are shared. Factory pattern enables config-driven solver selection.

## ADR-004: YAML + JSON Case Format
**Date**: Phase 1
**Decision**: Simulation definitions use YAML (`case.yaml`) with geometry either parametric (inline in YAML) or from JSON files. Pydantic models validate all config.
**Reason**: YAML is human-readable and supports comments. JSON is compact for raw node/panel data. Pydantic provides type safety and clear error messages for invalid configs.

## ADR-005: ProcessorPipeline with Topological Sort
**Date**: Phase 4
**Decision**: Post-processors declare `requires` and `produces` sets. `ProcessorPipeline` sorts them topologically before execution.
**Reason**: Eliminates manual ordering. Adding a new processor only needs its dependencies declared. Pipeline validates completeness before running.

## ADR-006: OpenFOAM Validation via foamlib
**Date**: Phase 3
**Decision**: Use `foamlib` (with fallback to `PyFoam`) for OpenFOAM case generation and manipulation rather than raw file editing.
**Reason**: foamlib handles OpenFOAM dictionary parsing, provides typed Python API for case setup, and manages parallel decomposition. Reduces string-manipulation bugs.

## ADR-007: Potential-Based Surface Velocity Recovery
**Date**: Phase 6
**Decision**: Compute surface velocity by differentiating the velocity potential along the surface ($V_t = d\phi/ds$) rather than summing tangential influence coefficients.
**Reason**: More robust at corners and high-curvature regions where direct influence summation can produce oscillations. Slightly more expensive but significantly more accurate for non-smooth geometries.

## ADR-008: Per-Component Body Masking
**Date**: Phase 5
**Decision**: `VelocityField2D` masks grid points inside each body component independently using the component's mesh boundary.
**Reason**: Prevents velocity artifacts inside solid bodies. Per-component masking handles multi-body cases where bodies may be close together, avoiding masking gaps between them.

## ADR-009: MeshBase/Mesh2D/Mesh3D Hierarchy
**Date**: 3D Solver Phase 0
**Decision**: Refactored the flat `Mesh` geometry container into `MeshBase` (ABC) with `Mesh2D` and `Mesh3D` implementations. Added native quad panel properties.
**Reason**: Ensures a unified API for data structures so solvers and components can be extended to 3D without polluting 2D namespaces or maintaining two identical architectural stacks. 

## ADR-010: Numba JIT and Parallelization for 3D
**Date**: 3D Solver Phase 1
**Decision**: Replaced all pure Python nested `for` loops in 3D source influence and velocity evaluations with `@njit(fastmath=True, cache=True)` compiled functions utilizing `prange` parallelization.
**Reason**: The 3D constant-strength source method requires evaluating computationally heavy expressions (logarithms, arctangents) repeatedly ($O(N^2)$ scaling). Initial tests found solving a ~2000 panel system timed out in pure Python. The Numba integration reduced influence matrix build times from >120s down to ~0.5s for 2000 panels, enabling rapid iteration and future higher-resolution meshes.
