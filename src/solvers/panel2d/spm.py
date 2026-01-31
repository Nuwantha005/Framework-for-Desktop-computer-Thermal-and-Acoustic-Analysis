
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from core.geometry.mesh import Mesh
from .base import PanelSolver2D, PanelMethodConfig
from .influences import compute_source_influence_matrices, compute_source_velocity_influence


class SourcePanelSolver(PanelSolver2D):
    """
    2D Constant-Strength Source Panel Method Solver.
    
    Implements the Katz & Plotkin formulation for source-only panels
    with constant strength distribution on flat line segments.
    
    Assumes pure potential flow (no lift, no wake).
    """
    
    def __init__(self, mesh: Mesh, v_inf: float = 1.0, aoa: float = 0.0):
        """
        Initialize the solver.
        
        Args:
            mesh: 2D panel mesh
            v_inf: Freestream velocity magnitude
            aoa: Angle of attack in DEGREES
        """
        super().__init__(mesh, v_inf, aoa)
        
        # Internal: singularity strengths
        self._sigma: NDArray[np.float64] = None
        self._influence_matrices: dict = None
    
    @property
    def config(self) -> PanelMethodConfig:
        """Return configuration for this solver."""
        return PanelMethodConfig(
            singularity_type="source",
            panel_order="constant",
            panel_geometry="flat"
        )
    
    @property
    def sigma(self) -> NDArray[np.float64]:
        """
        Source strengths for each panel.
        
        Returns:
            (N,) array of source strengths
        
        Raises:
            RuntimeError: If solver hasn't been executed yet
        """
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self._sigma
    
    # --- Implementation of abstract methods ---
    
    def _compute_influence_matrices(self) -> dict:
        """
        Compute geometric influence integrals for source panels.
        
        Returns:
            Dict with "I" (normal) and "J" (tangential) matrices
        """
        I, J = compute_source_influence_matrices(self._mesh)
        return {"I": I, "J": J}
    
    def _solve_linear_system(self, influence_matrices: dict) -> dict:
        """
        Solve A*sigma = b for source strengths.
        
        Args:
            influence_matrices: Dict with "I" matrix
        
        Returns:
            Dict with "source" key mapping to sigma array
        """
        I = influence_matrices["I"]
        n_panels = len(self._mesh.panels)
        
        # Assemble linear system: A * sigma = b
        A = I.copy()
        np.fill_diagonal(A, np.pi)
        
        # RHS: -V_inf * 2π * cos(beta)
        # where beta is angle between panel normal and freestream
        normals = self._mesh.normals[:, :2]  # (N, 2)
        v_inf_2d = self.v_inf_vector[:2]  # (2,)
        cos_beta = normals @ v_inf_2d / self._v_inf
        
        b = -self._v_inf * 2 * np.pi * cos_beta
        
        # Solve
        try:
            sigma = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("WARNING: Singular matrix in SourcePanelSolver. Check geometry.")
            sigma = np.zeros(n_panels)
        
        # Store internally
        self._sigma = sigma
        
        # Also store in mesh for backward compatibility
        self._mesh.cell_data['source_strength'] = sigma
        
        return {"source": sigma}
    
    def _compute_surface_velocity(
        self,
        influence_matrices: dict,
        strengths: dict
    ) -> NDArray[np.float64]:
        """
        Compute tangential velocity at panel centers.
        
        Vt = V_inf_tangential + sum(sigma_j * J_ij) / (2π)
        
        Args:
            influence_matrices: Dict with "J" matrix
            strengths: Dict with "source" key
        
        Returns:
            (N,) array of tangential velocity magnitudes
        """
        J = influence_matrices["J"]
        sigma = strengths["source"]
        
        # Freestream tangential component at each panel
        tangents = self._mesh.tangents[:, :2]  # (N, 2)
        v_inf_2d = self.v_inf_vector[:2]  # (2,)
        v_inf_tangential = tangents @ v_inf_2d
        
        # Induced tangential velocity from source panels
        induced_tangential = (J @ sigma) / (2 * np.pi)
        
        # Total tangential velocity
        Vt = v_inf_tangential + induced_tangential
        
        return Vt
    
    def _velocity_at_points(
        self,
        points: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute velocity induced by source panels at arbitrary points.
        
        This replaces the old VelocityField2D._compute_point_velocity().
        
        Args:
            points: (M, 3) array of (x, y, z) coordinates
        
        Returns:
            (Vx, Vy) tuple of (M,) arrays
        """
        # Extract panel geometry
        panel_start_indices = self._mesh.panels[:, 0]
        nodes_start = self._mesh.nodes[panel_start_indices, :2]
        panel_lengths = self._mesh.areas
        tangents = self._mesh.tangents[:, :2]
        phi = np.arctan2(tangents[:, 1], tangents[:, 0])
        phi = np.where(phi < 0, phi + 2*np.pi, phi)
        
        # Initialize velocity arrays
        num_points = len(points)
        Vx = np.zeros(num_points)
        Vy = np.zeros(num_points)
        
        # Compute induced velocity from each panel at each point
        for idx, point in enumerate(points):
            point_2d = point[:2]
            
            for j in range(len(self._mesh.panels)):
                # Compute influence coefficients
                Mx, My = compute_source_velocity_influence(
                    point=point_2d,
                    panel_start=nodes_start[j],
                    panel_length=panel_lengths[j],
                    panel_angle=phi[j]
                )
                
                # Add contribution from this panel
                Vx[idx] += self._sigma[j] * Mx / (2 * np.pi)
                Vy[idx] += self._sigma[j] * My / (2 * np.pi)
        
        # Add freestream velocity
        Vx += self.v_inf_vector[0]
        Vy += self.v_inf_vector[1]
        
        return Vx, Vy
    
    # --- Backward compatibility properties ---
    
    @property
    def sigma(self) -> NDArray[np.float64]:
        """Source strengths (backward compatibility)."""
        if self._sigma is None:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self._sigma
    
    @property
    def Vt(self) -> NDArray[np.float64]:
        """Tangential velocity at panel centers (backward compatibility)."""
        if not self.is_solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self.surface_velocity[:, 0]  # Extract tangential component
    
    @property
    def Cp(self) -> NDArray[np.float64]:
        """
        Pressure coefficient at panel centers (backward compatibility).
        
        Computed via Bernoulli: Cp = 1 - (V/V_inf)²
        """
        if not self.is_solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        
        # Compute total velocity magnitude at panel centers
        # For 2D on surface: Vn=0 (BC), so V_total = |Vt|
        Vt = self.surface_velocity[:, 0]  # Extract tangential component
        V_total = np.abs(Vt)
        
        # Bernoulli: Cp = 1 - (V/V_inf)²
        Cp = 1.0 - (V_total / self._v_inf)**2
        
        return Cp
