# 3D Panel Solver Implementation Plan

**Created**: 2026-03-31
**Status**: In Progress
**Branch**: `3d-panel-solver-implemention` (after Phase 0 on `main`)

## Overview

Extend the 2D panel method solver to support 3D geometries with:
- Quad surface panels (UV sphere topology)
- External mesh import (STL via meshio)
- VTK export for ParaView visualization
- Constant-source panel solver (Neumann BC)

## Key Decisions

- **Panel type**: Quads only (not triangles)
- **Mesh generation**: pygmsh for parametric (UV sphere), meshio for STL import
- **Visualization**: VTK file export for ParaView (not interactive PyVista)
- **Branch strategy**: Refactor mesh hierarchy on `main`, implement 3D solver on feature branch
- **Scope**: Vanilla panel solver first, no BL integration initially

## Reference Material

- K&P Chapter 12: Three-Dimensional Numerical Solutions
- K&P Chapter 10: 3D singularity element influence functions
- Located at: `notes_archived/Low-Speed Aerodynamics 2nd edition _Joseph Katz Allen Plotkin/chapter 12 Three-Dimensional Numerical Solutions.md`

## Branch Workflow

```
main ─────●──────────────●─────────────────────────────●───────
          │              │                             │
          │ Phase 0      │                             │ merge
          │ refactor     │                             │
          │              ▼                             │
          │    main (MeshBase/Mesh2D/Mesh3D)           │
          │                   │                        │
          │                   │ rebase                 │
          │                   ▼                        │
          │         3d-panel-solver-implemention ──●───┘
          │                              Phases 1-3
          │
          └─► boundary-layer-experimentation (can rebase later)
```

---

## Phase 0: Mesh Hierarchy Refactoring

**Branch**: `main`
**Commit message**: `refactor(geometry): introduce MeshBase/Mesh2D/Mesh3D hierarchy for 3D support`

### Files to Create

| File | Purpose |
|------|---------|
| `src/core/geometry/mesh_base.py` | `MeshBase` ABC with shared interface |
| `src/core/geometry/mesh3d.py` | `Mesh3D` for quad surface panels |
| `src/core/geometry/io/__init__.py` | Mesh I/O package |
| `src/core/geometry/io/gmsh_generator.py` | `generate_sphere()` via pygmsh |
| `src/core/geometry/io/stl_reader.py` | `read_stl()` via meshio |
| `src/core/geometry/io/vtk_export.py` | `export_solution_vtk()` |

### Files to Modify

| File | Change |
|------|--------|
| `src/core/geometry/mesh.py` | Rename class to `Mesh2D`, add `Mesh = Mesh2D` alias |
| `src/core/geometry/__init__.py` | Export `MeshBase`, `Mesh2D`, `Mesh3D` |

### MeshBase Interface

```python
class MeshBase(ABC):
    """Abstract base for 2D and 3D meshes."""
    
    nodes: NDArray[np.float64]           # (N, 3)
    panels: NDArray[np.int32]            # (P, k) where k=2 (2D) or 4 (3D)
    component_ids: NDArray[np.int32]     # (P,)
    
    centers: NDArray[np.float64]         # (P, 3)
    normals: NDArray[np.float64]         # (P, 3)
    areas: NDArray[np.float64]           # (P,)
    
    cell_data: Dict[str, NDArray]
    
    @property
    @abstractmethod
    def dimension(self) -> int: ...
    
    @abstractmethod
    def compute_geometry(self) -> None: ...
    
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_panels(self) -> int: ...
```

### Mesh2D Specifics

```python
class Mesh2D(MeshBase):
    """2D panel mesh with line segment panels."""
    
    panels: NDArray[np.int32]            # (P, 2)
    tangents: NDArray[np.float64]        # (P, 3)
    
    @property
    def dimension(self) -> int:
        return 2
```

### Mesh3D Specifics

```python
class Mesh3D(MeshBase):
    """3D surface mesh with quadrilateral panels."""
    
    panels: NDArray[np.int32]            # (P, 4)
    tangent1: NDArray[np.float64]        # (P, 3) - first tangent direction
    tangent2: NDArray[np.float64]        # (P, 3) - second tangent direction
    
    @property
    def dimension(self) -> int:
        return 3
```

### Validation

- [ ] All existing 2D tests pass
- [ ] `Mesh` alias works for backward compatibility
- [ ] `demos/demo_mesh.py` runs successfully
- [ ] `demos/demo_streamlines.py` runs successfully

---

## Phase 1: 3D Panel Solver

**Branch**: `3d-panel-solver-implemention`
**Commit message**: `feat(solvers): add 3D constant-source panel solver with quad panels`

### Files to Create

| File | Purpose |
|------|---------|
| `src/solvers/panel3d/__init__.py` | Package exports |
| `src/solvers/panel3d/base.py` | `PanelSolver3D` ABC |
| `src/solvers/panel3d/source3d.py` | `SourcePanelSolver3D` |
| `src/solvers/panel3d/influences/__init__.py` | Influence package |
| `src/solvers/panel3d/influences/source3d.py` | Quad source influence functions |

### Files to Modify

| File | Change |
|------|--------|
| `src/solvers/__init__.py` | Register 3D solver |
| `src/solvers/factory.py` | Add 3D solver support |

### Theory: Constant Source Quad Panel (Hess-Smith)

For a constant-strength source panel, the velocity potential at point P:

```
φ(P) = (σ/4π) ∫∫_S (1/r) dS
```

For planar quadrilateral panels, this integral has closed-form solutions.

**Velocity components** (K&P Eq. 10.22):
```
u = (σ/4π) ∫∫ (x - x₀)/r³ dS
v = (σ/4π) ∫∫ (y - y₀)/r³ dS  
w = (σ/4π) ∫∫ (z - z₀)/r³ dS
```

For implementation, decompose quad into triangles or use quad-specific formulas from Hess & Smith 1967.

### Solver Structure

```python
class SourcePanelSolver3D(Solver):
    """3D constant-strength source panel solver."""
    
    def __init__(self, mesh: Mesh3D, v_inf: float, aoa: float, aos: float = 0.0):
        """
        Args:
            mesh: 3D surface mesh
            v_inf: Freestream velocity magnitude
            aoa: Angle of attack (pitch) in degrees
            aos: Angle of sideslip (yaw) in degrees
        """
    
    def solve(self) -> None:
        """Solve Aσ = b for source strengths."""
    
    @property
    def surface_velocity(self) -> NDArray[np.float64]:
        """Velocity at panel centers (P, 3)."""
    
    def velocity_at(self, points: NDArray) -> NDArray[np.float64]:
        """Velocity at arbitrary points (M, 3)."""
    
    @property
    def Cp(self) -> NDArray[np.float64]:
        """Pressure coefficient at panel centers."""
```

### Validation

- [ ] Sphere Cp matches analytical: `Cp = 1 - 2.25*sin²θ`
- [ ] At equator (θ=90°): Cp = -1.25
- [ ] At stagnation (θ=0°): Cp = 1.0
- [ ] Error < 1% for fine mesh (1000+ panels)

---

## Phase 2: Mesh Generation & VTK Export

**Branch**: `3d-panel-solver-implemention`
**Commit message**: `feat(geometry): add sphere generation and VTK export for 3D`

### Sphere Generation via pygmsh

```python
def generate_sphere(
    n_theta: int = 16,
    n_phi: int = 32,
    radius: float = 1.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Mesh3D:
    """
    Generate UV sphere mesh with quad panels.
    
    Args:
        n_theta: Number of divisions in polar angle (latitude)
        n_phi: Number of divisions in azimuthal angle (longitude)
        radius: Sphere radius
        center: Center coordinates
    
    Returns:
        Mesh3D with (n_theta - 1) * n_phi quad panels
        (poles use degenerate quads or triangles)
    """
```

### STL Import

```python
def read_stl(path: str) -> Mesh3D:
    """
    Read STL file and convert to Mesh3D.
    
    Note: STL contains triangles; this function converts
    adjacent coplanar triangles to quads where possible,
    or raises error if quads required.
    """
```

### VTK Export

```python
def export_solution_vtk(
    mesh: Mesh3D,
    fields: Dict[str, NDArray],
    path: str
) -> None:
    """
    Export mesh with solution fields to VTK for ParaView.
    
    Args:
        mesh: 3D mesh
        fields: Dict of field name -> data array (Cp, velocity, etc.)
        path: Output path (.vtu or .vtk)
    """
```

### Demo Script

Create `demos/demo_sphere_3d.py`:

```python
"""3D sphere flow validation with VTK output."""

# 1. Generate sphere mesh
mesh = generate_sphere(n_theta=32, n_phi=64, radius=0.5)

# 2. Create and solve
solver = SourcePanelSolver3D(mesh, v_inf=1.0, aoa=0.0)
solver.solve()

# 3. Compare with analytical
theta = np.arccos(mesh.centers[:, 0] / np.linalg.norm(mesh.centers, axis=1))
Cp_analytical = 1 - 2.25 * np.sin(theta)**2
error = np.abs(solver.Cp - Cp_analytical)
print(f"Max Cp error: {np.max(error):.4f}")

# 4. Export to VTK
export_solution_vtk(mesh, {"Cp": solver.Cp}, "sphere_flow.vtu")
print("Saved: sphere_flow.vtu (open in ParaView)")
```

---

## Phase 3: Case System Extension

**Branch**: `3d-panel-solver-implemention`  
**Commit message**: `feat(io): extend case system for 3D parametric and external meshes`

### Schema Updates

```python
# In schemas.py

class GeometryConfig(BaseModel):
    type: Literal["circle", "rectangle", "rounded_rectangle", 
                  "sphere", "box", "external"]  # Add 3D types
    parameters: Optional[Dict[str, Any]] = None
    file: Optional[str] = None  # For external meshes
```

### Case YAML Examples

**Parametric sphere:**
```yaml
name: "Sphere Flow"
case_type: "parametric_3d"

freestream:
  velocity: [1.0, 0.0, 0.0]

components:
  - name: "sphere"
    geometry:
      type: "sphere"
      parameters:
        radius: 0.5
        center: [0.0, 0.0, 0.0]
    mesh_levels:
      - [16, 32]   # n_theta, n_phi
      - [32, 64]
      - [64, 128]

solver:
  type: "source_3d"

visualization:
  domain:
    x_range: [-2.0, 3.0]
    y_range: [-2.0, 2.0]
    z_range: [-2.0, 2.0]
  resolution: [100, 80, 80]

fluid:
  density: 1.225
```

**External STL:**
```yaml
name: "Complex Body"
case_type: "external_3d"

components:
  - name: "body"
    geometry:
      type: "external"
      file: "body.stl"

freestream:
  velocity: [10.0, 0.0, 0.0]

solver:
  type: "source_3d"
```

### CaseLoader Updates

```python
def _build_scene_3d(config, base_path, mesh_level_index=0):
    """Build Scene for 3D case."""
    for comp_config in config.components:
        if comp_config.geometry.type == "sphere":
            mesh = generate_sphere(**comp_config.geometry.parameters)
        elif comp_config.geometry.type == "external":
            mesh = read_stl(base_path / comp_config.geometry.file)
        # ...
```

### Validation Case

Create `cases/sphere_flow/`:
```
cases/sphere_flow/
├── case.yaml
└── out/           # (gitignored)
```

---

## Dependencies

Already in `requirements.txt`:
- `gmsh>=4.11.0`
- `pygmsh>=7.1.0`
- `meshio>=5.3.0`
- `pyvista>=0.42.0`

---

## Success Criteria

| Milestone | Criteria |
|-----------|----------|
| Phase 0 complete | All 2D demos run, `Mesh2D` works as before |
| Phase 1 complete | Sphere Cp error < 1% vs analytical at fine mesh |
| Phase 2 complete | VTK loads in ParaView, shows Cp contours |
| Phase 3 complete | `Case.create_solver()` works for sphere_flow case |

---

## Future Extensions (Not This Implementation)

- Actuator disk model for fans
- Higher-order 3D panels (linear source)
- 3D BL path extraction via surface streamlines
- Wake modeling for lifting bodies
- PyVista interactive visualization
