"""
Velocity field computation for 2D panel methods.
Handles grid generation and caching to avoid redundant calculations.

Refactored to use solver.velocity_at() interface.
"""

import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING
from numpy.typing import NDArray
from matplotlib import path

from core.geometry.mesh import Mesh

if TYPE_CHECKING:
    from solvers.base import Solver


class VelocityField2D:
    """
    Computes and caches velocity field for 2D panel method solutions.
    
    This class separates expensive grid computation from visualization,
    allowing multiple plots to reuse the same computed field.
    
    Delegates velocity computation to solver.velocity_at() method.
    """
    
    def __init__(self, 
                 solver: "Solver",
                 mesh: Optional[Mesh] = None):
        """
        Initialize velocity field calculator.
        
        Args:
            solver: Solved Solver instance with velocity_at() method
            mesh: Optional mesh override (uses solver's mesh if None)
        
        Raises:
            ValueError: If solver not solved yet
        """
        if not solver.is_solved:
            raise ValueError("Solver must be solved before creating VelocityField2D")
        
        self.solver = solver
        self.mesh = mesh if mesh is not None else solver.mesh
        
        if self.mesh.dimension != 2:
            raise ValueError("VelocityField2D requires a 2D mesh")
        
        # Build boundary paths for each component separately
        self.boundary_paths = self._build_component_paths(self.mesh)
        
        # Cached field data
        self._XX: Optional[NDArray] = None
        self._YY: Optional[NDArray] = None
        self._Vx: Optional[NDArray] = None
        self._Vy: Optional[NDArray] = None
        self._x_range: Optional[Tuple[float, float]] = None
        self._y_range: Optional[Tuple[float, float]] = None
        self._resolution: Optional[Tuple[int, int]] = None
    
    def compute(self,
                x_range: Tuple[float, float],
                y_range: Tuple[float, float],
                resolution: Tuple[int, int] = (100, 100),
                force: bool = False) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Compute velocity field on a grid (with caching).
        
        Args:
            x_range: (xmin, xmax) domain extent
            y_range: (ymin, ymax) domain extent
            resolution: (nx, ny) grid points
            force: If True, recompute even if cached
            
        Returns:
            (XX, YY, Vx, Vy): Meshgrid coordinates and velocity components
        """
        # Check if we can reuse cached data
        if not force and self._is_cached(x_range, y_range, resolution):
            print("Using cached velocity field.")
            return self._XX, self._YY, self._Vx, self._Vy
        
        # Compute new field
        nx, ny = resolution
        xmin, xmax = x_range
        ymin, ymax = y_range
        
        x_grid = np.linspace(xmin, xmax, nx)
        y_grid = np.linspace(ymin, ymax, ny)
        XX, YY = np.meshgrid(x_grid, y_grid)
        
        # Flatten grid to (N, 2) for solver
        points_flat = np.column_stack([XX.ravel(), YY.ravel()])
        
        print(f"Computing velocity field: {nx}×{ny} grid via solver.velocity_at()...")
        
        # Use solver's velocity_at method (returns (N, 3) array)
        velocities = self.solver.velocity_at(points_flat)
        Vx_flat = velocities[:, 0]
        Vy_flat = velocities[:, 1]
        
        # Mask interior points (check each component separately)
        is_inside = self._points_inside_any_body(points_flat)
        Vx_flat[is_inside] = np.nan
        Vy_flat[is_inside] = np.nan
        
        # Reshape to grid
        Vx = Vx_flat.reshape(ny, nx)
        Vy = Vy_flat.reshape(ny, nx)
        
        # Cache results
        self._XX = XX
        self._YY = YY
        self._Vx = Vx
        self._Vy = Vy
        self._x_range = x_range
        self._y_range = y_range
        self._resolution = resolution
        
        print(f"✓ Field computed and cached.")
        
        return XX, YY, Vx, Vy
    
    def _build_component_paths(self, mesh: Mesh) -> list:
        """
        Build separate boundary paths for each component in the mesh.
        
        Returns:
            List of matplotlib Path objects, one per component
        """
        component_ids = np.unique(mesh.component_ids)
        paths = []
        
        for comp_id in component_ids:
            # Get panel indices for this component
            comp_mask = mesh.component_ids == comp_id
            comp_panel_indices = np.where(comp_mask)[0]
            
            if len(comp_panel_indices) == 0:
                continue
            
            # Collect ordered nodes for this component
            # Panels are assumed to be in order around the body
            comp_nodes = []
            for panel_idx in comp_panel_indices:
                n1_idx = mesh.panels[panel_idx, 0]
                comp_nodes.append(mesh.nodes[n1_idx, :2])
            
            # Close the path by adding the last panel's end node
            last_panel_idx = comp_panel_indices[-1]
            n2_idx = mesh.panels[last_panel_idx, 1]
            comp_nodes.append(mesh.nodes[n2_idx, :2])
            
            comp_nodes = np.array(comp_nodes)
            paths.append(path.Path(comp_nodes))
        
        return paths
    
    def _points_inside_any_body(self, points: NDArray) -> NDArray:
        """
        Check if points are inside any of the component boundaries.
        
        Args:
            points: (N, 2) array of points to check
            
        Returns:
            Boolean array of shape (N,) - True if point is inside any body
        """
        is_inside = np.zeros(len(points), dtype=bool)
        
        for body_path in self.boundary_paths:
            is_inside |= body_path.contains_points(points)
        
        return is_inside
    
    def _is_cached(self,
                   x_range: Tuple[float, float],
                   y_range: Tuple[float, float],
                   resolution: Tuple[int, int]) -> bool:
        """Check if requested field matches cached data."""
        if self._XX is None:
            return False
        return (self._x_range == x_range and
                self._y_range == y_range and
                self._resolution == resolution)
    
    def get_cached(self) -> Optional[Tuple[NDArray, NDArray, NDArray, NDArray]]:
        """Return cached field data if available."""
        if self._XX is None:
            return None
        return self._XX, self._YY, self._Vx, self._Vy
    
    def clear_cache(self):
        """Clear cached field data to free memory."""
        self._XX = None
        self._YY = None
        self._Vx = None
        self._Vy = None
        self._x_range = None
        self._y_range = None
        self._resolution = None
