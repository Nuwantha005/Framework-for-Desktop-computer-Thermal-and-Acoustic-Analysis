"""
Velocity potential calculation.

For irrotational flow, velocity can be derived from a potential:
    V = ∇φ  →  u = ∂φ/∂x, v = ∂φ/∂y

This processor integrates velocity to get the potential field.
"""

from typing import Set, Optional
import numpy as np
from numpy.typing import NDArray
from scipy import integrate

from .pipeline import PostProcessor
from .fields import FieldData
from .fluid import FluidState


class VelocityPotentialProcessor(PostProcessor):
    """
    Compute velocity potential from velocity field.
    
    For irrotational flow: φ such that V = ∇φ
    
    Integration approach:
        φ(x,y) = φ_ref + ∫u·dx + ∫v·dy
    
    Since the flow should be irrotational (curl V = 0),
    the result should be path-independent.
    
    Produces:
        - velocity_potential: φ [m²/s]
        - stream_function: ψ [m²/s] (for 2D incompressible)
    """
    
    @property
    def requires(self) -> Set[str]:
        return {"velocity"}
    
    @property
    def produces(self) -> Set[str]:
        return {"velocity_potential", "stream_function"}
    
    def process(self, fields: FieldData, fluid: Optional[FluidState] = None) -> None:
        """Compute velocity potential and stream function."""
        vel = fields.get_vector("velocity")
        if vel is None:
            raise ValueError("Velocity field not found")
        
        u = vel.u
        v = vel.v
        XX = fields.XX
        YY = fields.YY
        
        ny, nx = u.shape
        dx = XX[0, 1] - XX[0, 0]
        dy = YY[1, 0] - YY[0, 0]
        
        # Velocity potential: dφ/dx = u, dφ/dy = v
        # Integrate along x first, then adjust with y
        phi = self._integrate_potential(u, v, dx, dy)
        
        # Stream function: dψ/dy = u, dψ/dx = -v (for 2D incompressible)
        # ψ = const along streamlines
        psi = self._integrate_stream_function(u, v, dx, dy)
        
        fields.add_scalar("velocity_potential", phi, units="m²/s")
        fields.add_scalar("stream_function", psi, units="m²/s")
    
    def _integrate_potential(self, u: NDArray, v: NDArray, 
                             dx: float, dy: float) -> NDArray:
        """
        Integrate velocity to get potential.
        
        φ(x,y) = ∫₀ˣ u(x',0)dx' + ∫₀ʸ v(x,y')dy'
        """
        ny, nx = u.shape
        phi = np.zeros_like(u)
        
        # Handle NaN values (inside bodies)
        u_clean = np.nan_to_num(u, nan=0.0)
        v_clean = np.nan_to_num(v, nan=0.0)
        
        # Integrate u along x (at y=0, i.e., first row)
        phi[0, :] = np.cumsum(u_clean[0, :]) * dx
        
        # For each column, integrate v along y
        for i in range(nx):
            phi[:, i] = phi[0, i] + np.cumsum(v_clean[:, i]) * dy
        
        # Restore NaN for body interior
        phi[np.isnan(u)] = np.nan
        
        return phi
    
    def _integrate_stream_function(self, u: NDArray, v: NDArray,
                                    dx: float, dy: float) -> NDArray:
        """
        Integrate to get stream function.
        
        ψ(x,y) = ∫₀ʸ u(0,y')dy' - ∫₀ˣ v(x',y)dx'
        
        For incompressible 2D: ∂ψ/∂y = u, ∂ψ/∂x = -v
        """
        ny, nx = u.shape
        psi = np.zeros_like(u)
        
        # Handle NaN values
        u_clean = np.nan_to_num(u, nan=0.0)
        v_clean = np.nan_to_num(v, nan=0.0)
        
        # Integrate u along y (at x=0, i.e., first column)
        psi[:, 0] = np.cumsum(u_clean[:, 0]) * dy
        
        # For each row, integrate -v along x
        for j in range(ny):
            psi[j, :] = psi[j, 0] - np.cumsum(v_clean[j, :]) * dx
        
        # Restore NaN for body interior
        psi[np.isnan(u)] = np.nan
        
        return psi


class VorticityProcessor(PostProcessor):
    """
    Compute vorticity from velocity field.
    
    For 2D: ω = ∂v/∂x - ∂u/∂y (z-component of curl)
    
    Produces:
        - vorticity: ω [1/s]
    """
    
    @property
    def requires(self) -> Set[str]:
        return {"velocity"}
    
    @property
    def produces(self) -> Set[str]:
        return {"vorticity"}
    
    def process(self, fields: FieldData, fluid: Optional[FluidState] = None) -> None:
        """Compute vorticity."""
        vel = fields.get_vector("velocity")
        if vel is None:
            raise ValueError("Velocity field not found")
        
        u = vel.u
        v = vel.v
        
        dx = fields.XX[0, 1] - fields.XX[0, 0]
        dy = fields.YY[1, 0] - fields.YY[0, 0]
        
        # Vorticity: ω = dv/dx - du/dy
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        
        omega = dvdx - dudy
        
        fields.add_scalar("vorticity", omega, units="1/s")
