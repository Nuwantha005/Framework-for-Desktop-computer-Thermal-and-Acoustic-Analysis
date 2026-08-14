# Geometry Module State
**Last modified**: 2026-04-01

## Files
- `primitives.py` — `Point3D`, `Vector3D` dataclasses; `rotation_matrix_z()`, `rotation_matrix_xyz()` helpers
- `mesh_base.py` — `MeshBase` ABC: common mesh interface for node properties, areas, centers, normals, and serialization methods (JSON/VTK).
- `mesh.py` — `Mesh2D` dataclass (`Mesh` alias): nodes `(N,3)`, panels `(N,2)`, computed geometry (centers, normals, tangents, areas), cell_data dict
- `mesh3d.py` — `Mesh3D` dataclass: nodes `(N,3)`, panels `(N,4)` quad faces, tangent1/tangent2, geometry computations
- `component.py` — `Transform` dataclass (translation + rotation matrix, factory methods), `Component` dataclass (name, local_mesh, transform, bc)
- `scene.py` — `Scene` dataclass: list of Components + freestream; `assemble()` merges into single `Mesh2D` or `Mesh3D` based on dimension
- `io/geometry_io.py` — `GeometryReader`, `GeometryWriter`: JSON mesh serialization, now supports `Mesh3D`.
- `io/gmsh_generator.py` — `generate_sphere()` via PyGmsh
- `io/stl_reader.py` — `read_stl()` via Meshio
- `io/vtk_export.py` — `export_solution_vtk()` exports directly to ParaView formats
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
