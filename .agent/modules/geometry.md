# Geometry Module State
**Last modified**: 2026-02-25

## Files
- `primitives.py` — `Point3D`, `Vector3D` dataclasses; `rotation_matrix_z()`, `rotation_matrix_xyz()` helpers
- `mesh.py` — `Mesh` dataclass: nodes `(N,3)`, panels `(N,2)`, computed geometry (centers, normals, tangents, areas), cell_data dict
- `component.py` — `Transform` dataclass (translation + rotation matrix, factory methods), `Component` dataclass (name, local_mesh, transform, bc)
- `scene.py` — `Scene` dataclass: list of Components + freestream; `assemble()` merges into single Mesh
- `generators.py` — `generate_circle(n, radius, center)`, `generate_rectangle(nx, ny, width, height, center)`, `generate_rounded_rectangle(n_side, n_arc, width, height, corner_radius, center)`
- `factory.py` — `GeometryFactory`: registry mapping type strings → generator functions; `create()`, `register()`, `list_types()`

## Public API
- `Mesh(nodes, panels, dimension, component_ids)` — call `compute_geometry()` to populate centers/normals/tangents/areas
- `Component(name, local_mesh, transform, bc_type)` → `get_global_mesh(component_id) -> Mesh`
- `Scene(name, components, freestream)` → `assemble() -> Mesh`, `get_component(name)`
- `GeometryFactory.create(geometry_definition, resolution) -> Mesh`

## Data Flow
Case YAML → `CaseLoader` reads geometry type/params → `GeometryFactory.create()` → `Mesh` → wrapped in `Component` with `Transform` → assembled into `Scene` → `Scene.assemble()` → single merged `Mesh` for solver

## Dependencies
- Internal: none (leaf module)
- External: numpy, numpy.typing, shapely (rounded_rectangle generator)

## What's Next
- 3D mesh support (quad panels, surface normals)
- STL import/export for 3D geometries
- Additional generators (ellipse, NACA airfoils)

## Known Issues
- `generate_rounded_rectangle` uses shapely for offset curves — could be pure numpy
