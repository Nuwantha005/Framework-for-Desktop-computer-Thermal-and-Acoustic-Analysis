"""
Surface data extraction for panel method solvers.

Provides utilities to extract and process surface quantities like
tangential velocity, pressure coefficient, and other boundary data.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from core.geometry.mesh import Mesh


@dataclass
class SurfaceData:
    """
    Container for surface flow quantities along body boundary.
    
    All arrays have length N (number of panels or sample points).
    """
    # Spatial coordinates
    x: NDArray[np.float64]              # (N,) x-coordinates
    y: NDArray[np.float64]              # (N,) y-coordinates
    s: Optional[NDArray[np.float64]]    # (N,) arc length from start
    
    # Flow quantities
    Vt: NDArray[np.float64]             # (N,) tangential velocity magnitude
    Vn: Optional[NDArray[np.float64]]   # (N,) normal velocity (should be ~0 for walls)
    Cp: NDArray[np.float64]             # (N,) pressure coefficient
    
    # Metadata
    component_id: Optional[NDArray[np.int32]] = None  # (N,) component ID for each point
    component_name: Optional[str] = None
    source: str = "panel_method"        # Data source identifier
    
    def __len__(self) -> int:
        """Number of surface points."""
        return len(self.x)
    
    def get_component_mask(self, comp_id: int) -> NDArray[np.bool_]:
        """Get boolean mask for specific component."""
        if self.component_id is None:
            return np.ones(len(self), dtype=bool)
        return self.component_id == comp_id
    
    def subset(self, mask: NDArray[np.bool_]) -> "SurfaceData":
        """Extract subset of surface data using boolean mask."""
        return SurfaceData(
            x=self.x[mask],
            y=self.y[mask],
            s=self.s[mask] if self.s is not None else None,
            Vt=self.Vt[mask],
            Vn=self.Vn[mask] if self.Vn is not None else None,
            Cp=self.Cp[mask],
            component_id=self.component_id[mask] if self.component_id is not None else None,
            component_name=self.component_name,
            source=self.source
        )


class SurfaceDataExtractor:
    """
    Extract surface flow quantities from panel method solver results.
    
    Provides methods to extract tangential velocity, pressure coefficient,
    and other surface quantities from solved panel method results.
    
    Usage:
        solver = SourcePanelSolver(mesh, v_inf=10.0, aoa=0.0)
        solver.solve()
        
        extractor = SurfaceDataExtractor(mesh, solver)
        surface_data = extractor.extract()
        
        # Plot Cp distribution
        import matplotlib.pyplot as plt
        plt.plot(surface_data.s, surface_data.Cp)
    """
    
    def __init__(self, mesh: Mesh, solver):
        """
        Initialize extractor.
        
        Args:
            mesh: Panel mesh
            solver: Solved panel method solver instance (must have Vt, Cp attributes)
        """
        self.mesh = mesh
        self.solver = solver
    
    def extract(
        self,
        arc_length: bool = True,
        component_id: Optional[int] = None
    ) -> SurfaceData:
        """
        Extract surface quantities from solver results.
        
        Args:
            arc_length: If True, compute arc length coordinate s
            component_id: If provided, extract only this component
        
        Returns:
            SurfaceData with surface flow quantities
        """
        # Get solver results
        if not hasattr(self.solver, 'Vt') or self.solver.Vt is None:
            raise ValueError("Solver must be solved before extracting surface data")
        
        Vt = self.solver.Vt
        Cp = self.solver.Cp
        
        # Get panel center coordinates
        centers = self.mesh.centers  # (N, 3)
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Component IDs
        comp_ids = None
        if hasattr(self.mesh, 'component_ids'):
            comp_ids = self.mesh.component_ids
        
        # Filter by component if requested
        if component_id is not None:
            if comp_ids is None:
                raise ValueError("Mesh does not have component_ids attribute")
            mask = comp_ids == component_id
            x = x[mask]
            y = y[mask]
            Vt = Vt[mask]
            Cp = Cp[mask]
            comp_ids = comp_ids[mask]
        
        # Compute arc length if requested
        s = None
        if arc_length:
            s = self._compute_arc_length(x, y)
        
        # Normal velocity (should be ~0 for Neumann BC)
        Vn = None  # Panel method enforces Vn=0, no need to store
        
        return SurfaceData(
            x=x,
            y=y,
            s=s,
            Vt=Vt,
            Vn=Vn,
            Cp=Cp,
            component_id=comp_ids,
            source="panel_method"
        )
    
    def extract_by_component(self, arc_length: bool = True) -> dict[int, SurfaceData]:
        """
        Extract surface data separately for each component.
        
        Args:
            arc_length: If True, compute arc length for each component
        
        Returns:
            Dictionary mapping component_id -> SurfaceData
        """
        if not hasattr(self.mesh, 'component_ids'):
            # Single component
            return {0: self.extract(arc_length=arc_length)}
        
        unique_ids = np.unique(self.mesh.component_ids)
        return {
            comp_id: self.extract(arc_length=arc_length, component_id=comp_id)
            for comp_id in unique_ids
        }
    
    def interpolate_to_arc_length(
        self,
        surface_data: SurfaceData,
        s_new: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Interpolate Vt and Cp to new arc length positions.
        
        Useful for comparing with external data at different sampling points.
        
        Args:
            surface_data: Surface data with arc length coordinate
            s_new: New arc length positions to interpolate to
        
        Returns:
            (Vt_interp, Cp_interp) at new positions
        """
        if surface_data.s is None:
            raise ValueError("Surface data must have arc length coordinate")
        
        Vt_interp = np.interp(s_new, surface_data.s, surface_data.Vt)
        Cp_interp = np.interp(s_new, surface_data.s, surface_data.Cp)
        
        return Vt_interp, Cp_interp
    
    @staticmethod
    def _compute_arc_length(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute cumulative arc length along curve.
        
        Args:
            x: x-coordinates (N,)
            y: y-coordinates (N,)
        
        Returns:
            Arc length s (N,), starting from 0
        """
        # Compute distances between consecutive points
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        
        # Cumulative sum (prepend 0 for start)
        s = np.concatenate([[0], np.cumsum(ds)])
        
        return s
    
    def compute_surface_forces(
        self,
        surface_data: SurfaceData,
        rho: float = 1.225,
        v_inf: float = 1.0
    ) -> dict:
        """
        Compute integrated surface forces from pressure distribution.
        
        Args:
            surface_data: Surface data with Cp
            rho: Fluid density (kg/m³)
            v_inf: Freestream velocity (m/s)
        
        Returns:
            dict with 'Fx', 'Fy', 'Cl', 'Cd' (2D per-unit-span)
        """
        # Pressure from Cp: p = p_inf + 0.5 * rho * v_inf^2 * Cp
        # Dynamic pressure
        q_inf = 0.5 * rho * v_inf**2
        
        # Get normals at panel centers
        normals = self.mesh.normals  # (N, 3)
        nx = normals[:, 0]
        ny = normals[:, 1]
        
        # Panel lengths (for integration)
        S = self.mesh.areas  # Actually lengths in 2D
        
        # Filter by component if needed
        if surface_data.component_id is not None:
            # This is already filtered, get corresponding panels
            # This is tricky - need to map back. For now, assume full mesh
            pass
        
        # Force per panel: F = -p * n * dS (pressure acts normal to surface)
        # Negative because pressure pushes inward (opposite to outward normal)
        Cp = surface_data.Cp
        Fx = -q_inf * Cp * nx * S
        Fy = -q_inf * Cp * ny * S
        
        # Total forces (sum over panels)
        Fx_total = np.sum(Fx)
        Fy_total = np.sum(Fy)
        
        # Lift and drag (assuming freestream is in +x direction)
        # For general AoA, need to rotate
        # L = -Fx*sin(aoa) + Fy*cos(aoa)
        # D = Fx*cos(aoa) + Fy*sin(aoa)
        # For now, assume aoa=0: L=Fy, D=Fx
        
        # Compute reference area (chord or characteristic length)
        # For 2D, use chord length (x-extent of first component)
        x_min = surface_data.x.min()
        x_max = surface_data.x.max()
        chord = x_max - x_min
        
        # Coefficients (per unit span)
        Cl = Fy_total / (q_inf * chord) if chord > 0 else 0
        Cd = Fx_total / (q_inf * chord) if chord > 0 else 0
        
        return {
            'Fx': Fx_total,
            'Fy': Fy_total,
            'Cl': Cl,
            'Cd': Cd,
            'chord': chord,
            'q_inf': q_inf
        }
