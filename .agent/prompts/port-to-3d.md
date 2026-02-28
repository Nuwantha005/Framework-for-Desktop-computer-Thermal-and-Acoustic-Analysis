# Prompt: Port to 3D Panel Methods with PyVista

## Context
Extend the panel method solver from 2D to 3D. The `(N, 3)` array convention already used
throughout the codebase was designed with this in mind. 3D visualization uses PyVista (VTK)
instead of matplotlib. This is the final major solver extension.

**Prerequisites**: 2D higher-order panels and BL solver should be complete before starting
3D work. However, the infrastructure (directory structure, Mesh3D, PyVista plumbing) can
be set up in advance.

## Reference Materials
- **3D panel design**: `notes_archived/solver_implementation/05_3d_panel_methods.md`
- **K&P Chapter 12**: `notes_archived/Low-Speed Aerodynamics 2nd edition*/chapter 12*.md`
- **Coupled solver notes**: `notes_archived/solver_implementation/06_coupled_solvers.md`
- **Target directory**: `src/solvers/panel3d/` (currently has README.md only)
- **PyVista**: Already a dependency in pyproject.toml

## Architecture

### Mesh3D
```python
@dataclass
class Mesh3D:
    """3D surface mesh for panel methods.
    
    Supports triangular and quadrilateral panels.
    """
    nodes: NDArray[np.float64]     # (V, 3) vertex coordinates
    panels: NDArray[np.int32]      # (N, 3) triangles or (N, 4) quads
    
    @property
    def centers(self) -> NDArray:   # (N, 3) panel centroids
    
    @property
    def normals(self) -> NDArray:   # (N, 3) outward unit normals
    
    @property
    def areas(self) -> NDArray:     # (N,) panel areas
    
    def to_pyvista(self) -> pv.PolyData:
        """Convert to PyVista mesh for visualization."""
```

### 3D Solver Hierarchy
```
src/solvers/panel3d/
├── __init__.py
├── base.py               # PanelSolver3D ABC
├── spm3d.py              # SourcePanelSolver3D (constant source)
├── spm3d_linear.py       # LinearSourcePanelSolver3D (if 2D linear works well)
└── influences/
    ├── __init__.py
    ├── source3d.py        # 3D source influence (Hess-Smith 1967)
    ├── doublet3d.py       # 3D doublet influence (solid angle)
    └── vortex3d.py        # 3D vortex influence (Biot-Savart)
```

### 3D Influence Coefficients
From K&P Chapter 12 and Hess-Smith (1967):

**Constant-strength source panel** (triangular):
$$\phi_j = \frac{\sigma_j}{4\pi} \int\int_{\Delta_j} \frac{1}{|\mathbf{r} - \mathbf{r}'|} \, dA'$$

Self-influence: $\phi_{ii} = -\sigma_i / 2$

**Implementation approach**:
- Analytical integration for flat triangular/quad panels (Hess-Smith formulas)
- Numerical quadrature for curved panels or higher-order distributions
- Vectorize using `np.einsum` for batch distance/cross-product operations

### PyVista Visualization
Replace matplotlib with PyVista for all 3D visualization:

```python
import pyvista as pv

class Visualizer3D:
    """3D visualization using PyVista/VTK."""
    
    def plot_mesh(self, mesh: Mesh3D, **kwargs) -> pv.Plotter:
        """Render surface mesh with panel edges."""
    
    def plot_scalar_field(self, mesh: Mesh3D, values: NDArray, 
                          name: str, **kwargs) -> pv.Plotter:
        """Color-map scalar field (Cp, Vt, etc.) on surface."""
    
    def plot_streamlines(self, mesh: Mesh3D, solver, **kwargs) -> pv.Plotter:
        """3D streamlines from velocity field."""
    
    def plot_comparison(self, panel_data, reference_data, **kwargs) -> pv.Plotter:
        """Side-by-side 3D comparison."""
```

### File Structure for 3D Visualization
```
src/visualization/
├── ... (existing 2D files)
├── visualizer3d.py        # Visualizer3D facade
├── field3d.py             # VelocityField3D (3D grid computation)
└── pyvista_utils.py       # Mesh3D → PolyData conversion, colormaps
```

## Implementation Phases

### Phase 1: Infrastructure (can do now)
- [ ] `Mesh3D` dataclass with PyVista conversion
- [ ] `PanelSolver3D` ABC extending `Solver`
- [ ] `Visualizer3D` skeleton with mesh plotting
- [ ] STL/OBJ import for Mesh3D (meshio already available)
- [ ] Unit tests for Mesh3D geometry computations

### Phase 2: Constant-Source 3D Solver
- [ ] `source3d.py` — Hess-Smith analytical influence coefficients for triangular panels
- [ ] `SourcePanelSolver3D` — solve + field velocity
- [ ] Validate on sphere (analytical Cp = 1 - 9/4 sin²θ)
- [ ] PyVista Cp contour plots on sphere surface

### Phase 3: Higher-Order + Validation
- [ ] Linear-strength 3D source panels (if 2D linear proves valuable)
- [ ] Validate on ellipsoid (semi-analytical solution)
- [ ] Validate on box/rectangular prism (compare with 2D results at mid-plane)
- [ ] OF comparison for 3D cases

### Phase 4: 3D BL + Thermal
- [ ] Extend BL solver for 3D: surface-following coordinates
- [ ] 3D thermal solver (BDIM extension)
- [ ] Computer component geometry (actual use case)

## Geometry Sources for 3D
- **Sphere**: Parametric generation (extend existing `generators.py` pattern)
- **Ellipsoid**: Parametric generation
- **Box/cube**: Extend `generate_rectangle` to 3D
- **Computer components**: Import from CAD via STL (meshio)
- **gmsh**: Already a dependency — use for 3D meshing from primitives

## Validation Cases
1. **Sphere**: Analytical solution available, primary validation benchmark
2. **Ellipsoid**: Semi-analytical (Lamb), tests non-uniform curvature
3. **Cube**: Compare with 2D single_square at mid-plane
4. **Actual component**: Real computer component CAD geometry (final application)

## Critical Rules for 3D
- Keep `(N, 3)` array convention — already natural for 3D
- Use PyVista (not matplotlib) for ALL 3D visualization
- Store 3D meshes as VTK/VTP files (not JSON) for efficiency
- Parallelize influence computation (3D matrices are larger: O(N²) with N=10,000+)
- Consider sparse matrix storage if N > 5000

## Post-Implementation
- Update `.agent/modules/solver.md` and `visualization.md`
- Add 3D case YAML schema to `src/core/config/schemas.py`
- Create demo: `demos/demo_3d_sphere.py`
- Update `.agent/PROJECT_CONTEXT.md`
