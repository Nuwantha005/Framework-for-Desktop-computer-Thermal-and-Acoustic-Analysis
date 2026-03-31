# ADR-002: 3D Panel Solver Architecture

**Date**: 2026-03-31
**Status**: Accepted
**Deciders**: User + Agent

## Context

The project needs to extend from 2D panel methods to 3D. Several architectural choices need to be made:

1. How to structure the mesh hierarchy (2D vs 3D)
2. What panel element type to support (triangles, quads, or both)
3. How to generate/import 3D meshes
4. How to visualize 3D results
5. Whether to refactor existing code or start fresh

## Decision

### 1. Mesh Hierarchy: Refactor to `MeshBase`/`Mesh2D`/`Mesh3D`

**Choice**: Create abstract base class with dimension-specific subclasses

**Rationale**:
- Cleaner abstraction than dimension flag in single class
- Shared interface (nodes, panels, normals, areas, cell_data)
- Type safety — solvers can require specific mesh type
- Backward compatibility via `Mesh = Mesh2D` alias

**Alternatives considered**:
- Single `Mesh` class with dimension flag (current state) — less type-safe
- Completely separate classes — code duplication

### 2. Panel Element Type: Quads Only

**Choice**: Support only quadrilateral panels for 3D

**Rationale**:
- UV sphere naturally produces quads
- Simpler influence coefficient implementation (one formula)
- Matches structured mesh paradigm
- Most external meshes can be converted to quads

**Alternatives considered**:
- Triangles only — more common in STL, but less regular
- Mixed tri/quad — more complex implementation

### 3. Mesh Generation: pygmsh for Parametric, meshio for Import

**Choice**: 
- `pygmsh` for UV sphere and other parametric shapes
- `meshio` for STL/UNV/MSH import

**Rationale**:
- pygmsh provides clean Python API for gmsh
- meshio is universal mesh I/O (already in deps)
- Separates generation from import concerns

### 4. Visualization: VTK Export for ParaView

**Choice**: Export `.vtu` files, user views in ParaView

**Rationale**:
- ParaView is more capable than inline PyVista
- Decouples solver from visualization
- VTK is standard format for CFD
- Can still use PyVista for export (it wraps VTK)

**Alternatives considered**:
- Interactive PyVista — more complex, less standard
- matplotlib 3D — not suitable for surface meshes

### 5. Branching: Refactor on `main`, Implement on Feature Branch

**Choice**: 
- Phase 0 (mesh refactoring) → `main`
- Phases 1-3 (3D solver) → `3d-panel-solver-implemention`

**Rationale**:
- Mesh refactoring benefits all branches
- Keeps main stable and improved
- Feature branch starts from solid foundation
- Other branches (BL experimentation) can rebase to get improvements

## Consequences

### Positive
- Clean separation of 2D and 3D code
- Shared infrastructure (case system, postprocessing concepts)
- BL solver will work on paths extracted from either 2D or 3D
- Main branch gets cleaner abstractions

### Negative
- Requires updating imports across codebase
- Two-phase merge (main first, then feature branch)
- Quad-only limits some external mesh formats

### Risks
- STL files are triangle-based; may need triangle→quad conversion
- pygmsh API may change between versions
- VTK format versions can vary

## Implementation Notes

- Keep `Mesh` as alias for `Mesh2D` for backward compatibility
- Use `dimension` property (not attribute) for duck-typing flexibility
- 3D solver does not need lifting body support initially (non-lifting bluff bodies only)
- Actuator disk model is separate future phase

## References

- K&P Chapter 10: 3D singularity elements
- K&P Chapter 12: Three-dimensional numerical solutions
- Hess & Smith (1967): Original 3D panel method formulation
