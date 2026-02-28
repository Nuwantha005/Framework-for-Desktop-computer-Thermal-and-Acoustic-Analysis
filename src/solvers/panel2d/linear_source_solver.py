"""
Linear-strength source panel method solver for 2D flow.

Implements a higher-order panel method with linear variation of source strength
along each flat panel. Strengths are continuous at nodes (N+1 unknowns).
For N panels on a closed body, there are N nodes.
"""

from typing import Dict, Tuple
import numpy as np
from numpy.typing import NDArray

from .base import PanelSolver2D, PanelMethodConfig
from .influences.linear_source import (
    compute_linear_source_influence_matrices,
    compute_linear_source_velocity_field
)


class LinearSourcePanelSolver(PanelSolver2D):
    """
    2D linear-strength source panel method solver.
    """
    
    @property
    def config(self) -> PanelMethodConfig:
        return PanelMethodConfig("source", "linear", "flat")

    # ── backward-compat properties (match SourcePanelSolver API) ────────

    @property
    def sigma(self) -> NDArray[np.float64]:
        """Node source strengths."""
        if self._sigma is None:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self._sigma

    @property
    def Vt(self) -> NDArray[np.float64]:
        """Tangential velocity magnitude at panel centres (N,)."""
        if not self.is_solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self.surface_velocity[:, 0]

    @property
    def Cp(self) -> NDArray[np.float64]:
        """Pressure coefficient via Bernoulli: Cp = 1 - (Vt/V_inf)**2."""
        if not self.is_solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        Vt = self.surface_velocity[:, 0]
        return 1.0 - (Vt / self._v_inf) ** 2
    
    def _compute_influence_matrices(self) -> Dict[str, NDArray]:
        """
        Compute I and J matrices for linear source panels.
        """
        I, J = compute_linear_source_influence_matrices(self._mesh)
        return {"I": I, "J": J}
    
    def _solve_linear_system(self, influence_matrices: Dict[str, NDArray]) -> Dict[str, NDArray]:
        """
        Solve continuous linear system for node source strengths.
        """
        I = influence_matrices["I"]
        
        # In the linear_source.py, I is the true velocity coefficient (divided by 2pi)
        # So we just say A = I, and RHS = -V_inf . n
        A = I.copy()
        
        # Freestream velocity vector (only x, y needed for 2D)
        v_inf_2d = self.v_inf_vector[:2]
        normals = self._mesh.normals[:, :2]
        
        # b = -V \cdot n
        b = -np.dot(normals, v_inf_2d)
        
        # Solve the system
        try:
            # We use lstsq because if multiple bodies are perfectly closed, 
            # A could be marginally singular if we don't fix the zero net mass addition explicitly.
            # But the linear source naturally satisfies continuity, and the problem is well-posed
            # (N equations at panel centers, N unknowns at nodes per body).
            sigma, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # If there's an issue with exactness, fallback to direct solve
            if rank == A.shape[1]:
                sigma = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            raise RuntimeError("Failed to solve linear system for source strengths (singular matrix).")
        
        self._sigma = sigma
        self._mesh.cell_data['source_strength'] = sigma
        
        return {"source": sigma}
    
    def _compute_surface_velocity(self, influence_matrices: Dict[str, NDArray], strengths: Dict[str, NDArray]) -> NDArray:
        """
        Compute tangential surface velocity along panels (at centers).
        """
        J = influence_matrices["J"]
        sigma = strengths["source"]
        
        v_inf_2d = self.v_inf_vector[:2]
        tangents = self._mesh.normals.copy()
        # the mesh tangent is (dx/ds, dy/ds)
        # However, to be consistent with how J was built:
        mesh_tangents = self._mesh.tangents[:, :2]
        
        # V_inf \cdot t
        v_inf_t = np.dot(mesh_tangents, v_inf_2d)
        
        # induced tangential velocity
        # Note: J already contains the true velocity coefficients.
        v_induced_t = J @ sigma
        
        Vt = v_inf_t + v_induced_t
        
        return Vt
    
    def _velocity_at_points(self, points: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Compute velocity vectors at arbitrary points.
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)
            
        strengths = self._sigma
            
        V_induced_2d = compute_linear_source_velocity_field(
            points, self._mesh, strengths
        )
        
        v_inf_2d = self.v_inf_vector[:2]
        V_tot_2d = v_inf_2d + V_induced_2d
        
        return V_tot_2d[:, 0], V_tot_2d[:, 1]

    def _compute_induced_normal_velocity(self) -> NDArray[np.float64]:
        """Compute induced normal velocity from source panels."""
        if self._influence_matrices is None or 'I' not in self._influence_matrices:
            raise RuntimeError("Influence matrices not available")
        
        I = self._influence_matrices['I']
        strengths = self._sigma
        
        # Induced normal velocity
        # note: I matrix handles both the integral and self-influence directly
        # because we added 0.25 to the self-panel nodes in linear_source.py
        # Also note that linear_source.py outputs the final coefficient directly (unlike spm where it is divided by 2pi later)
        Vn = I @ strengths
        
        return Vn
    
    def _perform_solver_specific_validation(self, validation_dir, show_plots: bool) -> dict:
        """Linear Source panel validation: node strength distribution statistics."""
        print("\n--- Linear Source Panel Method Validation ---")
        
        strengths = self._sigma
        sigma_sum = np.sum(strengths)
        sigma_max = np.max(strengths)
        sigma_min = np.min(strengths)
        sigma_mean = np.mean(strengths)
        sigma_std = np.std(strengths)
        
        print(f"Source strength statistics (Nodes):")
        print(f"  Sum σ:     {sigma_sum:.2e} ")
        print(f"  Min σ:     {sigma_min:.6f}")
        print(f"  Max σ:     {sigma_max:.6f}")
        print(f"  Mean σ:    {sigma_mean:.6f}")
        print(f"  Std σ:     {sigma_std:.6f}")
        
        return {
            "sum_sigma": sigma_sum,
            "max_sigma": sigma_max,
            "min_sigma": sigma_min,
            "mean_sigma": sigma_mean,
            "std_sigma": sigma_std
        }
