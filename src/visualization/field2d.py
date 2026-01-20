"""
Velocity field computation for 2D panel methods.
Handles grid generation and caching to avoid redundant calculations.
"""

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray
from multiprocessing import Pool
from matplotlib import path

from core.geometry.mesh import Mesh


def _compute_point_velocity(args):
    """
    Worker function for parallel velocity computation at a single point.
    Must be module-level for multiprocessing pickling.
    
    Args:
        args: Tuple of (XP, YP, sigma, nodes_X, nodes_Y, phi, S, v_inf, aoa_rad)
        
    Returns:
        (Vx, Vy): Velocity components at point P
    """
    XP, YP, sigma, nodes_X, nodes_Y, phi, S, v_inf, aoa_rad = args
    
    # Vectorized computation over all panels
    dx = XP - nodes_X
    dy = YP - nodes_Y
    
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    A = -dx * cos_phi - dy * sin_phi
    B = np.maximum(dx**2 + dy**2, 1e-12)
    
    E_sq = np.maximum(B - A**2, 0)
    E = np.sqrt(E_sq)
    
    # Influence coefficients (Katz & Plotkin formulation)
    Cx = -cos_phi
    Dx = dx
    Cy = -sin_phi
    Dy = dy
    
    log_term = np.log(np.maximum((S**2 + 2*A*S + B) / B, 1e-12))
    
    E_safe = np.where(E < 1e-12, 1.0, E)
    atan_term = np.arctan2((S + A), E_safe) - np.arctan2(A, E_safe)
    
    term2_factor = 1.0 / E_safe
    
    Mx = 0.5 * Cx * log_term + ((Dx - A*Cx) * term2_factor) * atan_term
    My = 0.5 * Cy * log_term + ((Dy - A*Cy) * term2_factor) * atan_term
    
    # Zero out singular points
    mask = E < 1e-12
    Mx[mask] = 0
    My[mask] = 0
    
    # Sum panel contributions
    u_induced = np.sum(sigma * Mx) / (2 * np.pi)
    v_induced = np.sum(sigma * My) / (2 * np.pi)
    
    Vx = v_inf * np.cos(aoa_rad) + u_induced
    Vy = v_inf * np.sin(aoa_rad) + v_induced
    
    return Vx, Vy


class VelocityField2D:
    """
    Computes and caches velocity field for 2D panel method solutions.
    
    This class separates expensive grid computation from visualization,
    allowing multiple plots to reuse the same computed field.
    """
    
    def __init__(self, 
                 mesh: Mesh,
                 v_inf: float,
                 aoa: float,
                 source_strengths: NDArray):
        """
        Initialize velocity field calculator.
        
        Args:
            mesh: 2D mesh geometry
            v_inf: Freestream velocity magnitude
            aoa: Angle of attack (degrees)
            source_strengths: Panel source strengths (sigma)
        """
        if mesh.dimension != 2:
            raise ValueError("VelocityField2D requires a 2D mesh")
            
        self.mesh = mesh
        self.v_inf = v_inf
        self.aoa_rad = np.radians(aoa)
        self.sigma = source_strengths
        
        # Extract panel geometry for workers
        panel_start_indices = mesh.panels[:, 0]
        self.nodes_X = mesh.nodes[panel_start_indices, 0]
        self.nodes_Y = mesh.nodes[panel_start_indices, 1]
        self.S = mesh.areas
        
        # Panel angles
        tx = mesh.tangents[:, 0]
        ty = mesh.tangents[:, 1]
        self.phi = np.arctan2(ty, tx)
        self.phi = np.where(self.phi < 0, self.phi + 2*np.pi, self.phi)
        
        # Build boundary paths for each component separately
        self.boundary_paths = self._build_component_paths(mesh)
        
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
                num_cores: int = 6,
                force: bool = False) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Compute velocity field on a grid (with caching).
        
        Args:
            x_range: (xmin, xmax) domain extent
            y_range: (ymin, ymax) domain extent
            resolution: (nx, ny) grid points
            num_cores: Parallel worker count
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
        
        points_flat = np.column_stack([XX.ravel(), YY.ravel()])
        n_points = len(points_flat)
        
        # Build task list for workers
        tasks = [
            (points_flat[i, 0], points_flat[i, 1],
             self.sigma, self.nodes_X, self.nodes_Y,
             self.phi, self.S, self.v_inf, self.aoa_rad)
            for i in range(n_points)
        ]
        
        print(f"Computing velocity field: {nx}×{ny} grid, {num_cores} cores...")
        
        with Pool(processes=num_cores) as pool:
            results = pool.map(_compute_point_velocity, tasks)
        
        results_arr = np.array(results)
        Vx_flat = results_arr[:, 0]
        Vy_flat = results_arr[:, 1]
        
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
