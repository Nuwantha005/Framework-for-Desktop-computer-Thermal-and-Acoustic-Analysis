"""
3D constant-strength source panel method solver.

Implements the source panel method for 3D non-lifting bodies (spheres, 
fuselages, etc.) following Katz & Plotkin Chapter 10 formulation.
"""

import numpy as np
from numpy.typing import NDArray

from core.geometry.mesh3d import Mesh3D
from .base import PanelSolver3D
from .influences import compute_source_influence_matrix, compute_all_velocities_influence


class SourcePanelSolver3D(PanelSolver3D):
    """
    3D Constant-Strength Source Panel Method Solver.
    
    Solves for source strengths σ on quad panels such that the Neumann
    boundary condition (V·n = 0) is satisfied at panel centers.
    
    Suitable for:
    - Non-lifting bodies (spheres, ellipsoids, fuselages)
    - Symmetric flows (no lift/circulation)
    
    Validation:
    - Sphere: Cp = 1 - 2.25*sin²θ (should match within 1% for fine mesh)
    """
    
    def __init__(
        self,
        mesh: Mesh3D,
        v_inf: NDArray[np.float64],
    ):
        """
        Initialize 3D source panel solver.
        
        Args:
            mesh: 3D surface mesh with quad panels
            v_inf: Freestream velocity vector (3,) - [Vx, Vy, Vz]
        """
        super().__init__(mesh, v_inf)
    
    def _compute_influence_matrix(self) -> NDArray[np.float64]:
        """
        Compute source influence matrix for normal velocity BC.
        
        Returns:
            (N, N) matrix where A[i,j] = (V induced by σ_j=1) · n_i
        """
        A = compute_source_influence_matrix(
            centers=self._mesh.centers,
            normals=self._mesh.normals,
            vertices=self._mesh.nodes,
            panels=self._mesh.panels,
        )
        return A
    
    def _solve_linear_system(self, influence_matrix: NDArray[np.float64]) -> dict:
        """
        Solve Aσ = b for source strengths.
        
        Boundary condition: V·n = 0
        => (V_∞ + V_induced)·n = 0
        => Σ_j A_ij σ_j = -V_∞·n_i
        
        Args:
            influence_matrix: (N, N) from _compute_influence_matrix()
        
        Returns:
            Dict with "source" key mapping to sigma array
        """
        n_panels = self._mesh.num_panels
        normals = self._mesh.normals
        
        # RHS: -V_∞ · n for each panel
        b = -np.einsum('ij,j->i', normals, self._v_inf)
        
        # Solve linear system
        try:
            sigma = np.linalg.solve(influence_matrix, b)
        except np.linalg.LinAlgError:
            print("WARNING: Singular matrix in SourcePanelSolver3D. Check geometry.")
            sigma = np.zeros(n_panels)
        
        # Store internally
        self._sigma = sigma
        
        return {"source": sigma}
    
    def _compute_surface_velocity(self, strengths: dict) -> NDArray[np.float64]:
        """
        Compute velocity at panel centers.
        
        For source panels, the tangential velocity is continuous across
        the panel, but the normal velocity has a jump of σ/2.
        
        Surface velocity = V_∞ + V_induced (with V·n = 0 by construction)
        
        Args:
            strengths: Dict with "source" key
        
        Returns:
            (N, 3) velocity vectors at panel centers
        """
        n_panels = self._mesh.num_panels
        sigma = strengths["source"]
        
        # Initialize with freestream
        velocity = np.tile(self._v_inf, (n_panels, 1))
        
        # Add induced velocity from all panels at all panel centers
        v_induced = compute_all_velocities_influence(
            points=self._mesh.centers,
            vertices=self._mesh.nodes,
            panels=self._mesh.panels,
            sigma=sigma,
        )
        velocity += v_induced
        
        # Project out normal component (should be ~0 already, but ensure)
        normals = self._mesh.normals
        v_normal = np.einsum('ij,ij->i', velocity, normals)
        velocity -= np.outer(v_normal, np.ones(3)) * normals
        
        return velocity
    
    def _velocity_at_points(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute velocity at arbitrary field points.
        
        Args:
            points: (M, 3) array of coordinates
        
        Returns:
            (M, 3) array of velocity vectors
        """
        n_points = points.shape[0]
        velocity = np.zeros((n_points, 3), dtype=np.float64)
        
        # Induced velocity for all points
        v_induced = compute_all_velocities_influence(
            points=points,
            vertices=self._mesh.nodes,
            panels=self._mesh.panels,
            sigma=self._sigma,
        )
        
        # Total = freestream + induced
        velocity = self._v_inf + v_induced
        
        return velocity
    
    def validate_sphere(self, radius: float = 1.0) -> dict:
        """
        Validate solver against analytical sphere Cp.
        
        For potential flow over a sphere:
        Cp = 1 - 2.25 * sin²θ
        
        where θ is the polar angle from the stagnation point.
        
        Args:
            radius: Sphere radius (for computing θ from center positions)
        
        Returns:
            Dict with error metrics
        """
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        
        # Compute polar angle θ for each panel center
        # θ = 0 at stagnation point (upstream), θ = π/2 at equator
        centers = self._mesh.centers
        
        # Freestream direction (normalized)
        v_dir = self._v_inf / self.v_inf_magnitude
        
        # For each center, compute angle from freestream axis
        # cos(θ) = (center · v_dir) / |center|
        center_magnitudes = np.linalg.norm(centers, axis=1)
        cos_theta = np.einsum('ij,j->i', centers, v_dir) / center_magnitudes
        sin_theta_sq = 1 - cos_theta**2
        
        # Analytical Cp
        Cp_analytical = 1 - 2.25 * sin_theta_sq
        
        # Computed Cp
        Cp_computed = self.Cp
        
        # Error metrics
        error = Cp_computed - Cp_analytical
        
        metrics = {
            "Cp_max_error": float(np.max(np.abs(error))),
            "Cp_rms_error": float(np.sqrt(np.mean(error**2))),
            "Cp_mean_error": float(np.mean(error)),
            "L_inf_error": float(np.max(np.abs(error))),
            "L2_error": float(np.sqrt(np.mean(error**2))),
            "Cp_analytical_range": [float(np.min(Cp_analytical)), float(np.max(Cp_analytical))],
            "Cp_computed_range": [float(np.min(Cp_computed)), float(np.max(Cp_computed))],
        }
        
        # Also store for visualization
        self._mesh.cell_data['Cp_analytical'] = Cp_analytical
        self._mesh.cell_data['Cp_error'] = error
        
        return metrics
