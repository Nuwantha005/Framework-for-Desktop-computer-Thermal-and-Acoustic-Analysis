"""
Constant-strength source panel method solver.

Implements the 2D source panel method following Katz & Plotkin's formulation
for constant-strength source distributions on flat line segments.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from pathlib import Path

from core.geometry.mesh import Mesh
from .base import PanelSolver2D, PanelMethodConfig
from .influences import compute_source_influence_matrices, compute_source_velocity_influence,compute_source_potential_influence


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
    
    def _compute_surface_velocity_from_summing_influences(
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
    
    def _compute_surface_velocity(
        self,
        influence_matrices: dict,
        strengths: dict
    ) -> NDArray[np.float64]:
        """
        Compute tangential velocity at panel centers using potential gradient.
    
        This method is more robust than direct velocity influence summation,
        especially at corners where curvature changes rapidly.
        """
        # Use potential-based approach
        Vt = self._compute_surface_velocity_from_potential()
        # Vt = self._compute_surface_velocity_from_summing_influences(influence_matrices, strengths);
    
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
    
    # --- Validation implementation ---
    
    def _compute_induced_normal_velocity(self) -> NDArray[np.float64]:
        """Compute induced normal velocity from source panels."""
        if self._influence_matrices is None or 'I' not in self._influence_matrices:
            raise RuntimeError("Influence matrices not available")
        
        I = self._influence_matrices['I']
        
        # Induced normal velocity: (I @ sigma) / (2π)
        Vn_induced = (I @ self._sigma) / (2 * np.pi)
        
        # Add self-influence: For source panels, self-influence is σ/2
        Vn_self = self._sigma / 2
        
        return Vn_induced + Vn_self
    
    def _perform_solver_specific_validation(self, validation_dir: Path, show_plots: bool) -> dict:
        """Source panel validation: mass conservation + σ distribution plot."""
        print("\n--- Source Panel Method Validation ---")
        
        # Check mass conservation: Σσ = 0 for closed bodies
        sigma_sum = np.sum(self._sigma)
        sigma_max = np.max(self._sigma)
        sigma_min = np.min(self._sigma)
        sigma_mean = np.mean(self._sigma)
        sigma_std = np.std(self._sigma)
        
        print(f"Source strength statistics:")
        print(f"  Sum σ:     {sigma_sum:.2e} ")
        print(f"  Min σ:     {sigma_min:.6f}")
        print(f"  Max σ:     {sigma_max:.6f}")
        print(f"  Mean σ:    {sigma_mean:.6f}")
        print(f"  Std σ:     {sigma_std:.6f}")
        
        # Plot σ distribution using surface_envelope
        self._plot_source_distribution(validation_dir, show_plots)
        
        return {
            "sigma_sum": float(sigma_sum),
            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
            "sigma_mean": float(sigma_mean),
            "sigma_std": float(sigma_std),
            "mass_conservation_error": float(sigma_sum)
        }
    
    def _plot_source_distribution(self, validation_dir: Path, show_plots: bool):
        """Plot source strength distribution using surface envelope."""
        from visualization.surface_envelope import plot_surface_envelope
        import matplotlib.pyplot as plt
        
        # Get surface coordinates
        centers = self._mesh.centers[:, :2]
        x, y = centers[:, 0], centers[:, 1]
        
        # Create plot
        fig, ax = plot_surface_envelope(
            x, y, self._sigma,
            scale=0.3,
            quantity_name="σ (source strength)",
            colormap='RdBu_r',
            title=f"Source Strength Distribution (N={self._mesh.num_panels} panels)",
            show_colorbar=True
        )
        
        # Save plot
        output_file = validation_dir / "source_strength_distribution.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Source distribution plot saved: {output_file}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def _compute_surface_potential(self) -> NDArray[np.float64]:
        """
        Compute total velocity potential at each panel center.
    
        Returns:
            (N,) array of potential values at panel centers
        """
        n_panels = len(self._mesh.panels)
        centers = self._mesh.centers[:, :2]
    
        # Get panel endpoints
        panel_start_indices = self._mesh.panels[:, 0]
        panel_end_indices = self._mesh.panels[:, 1]
        nodes_start = self._mesh.nodes[panel_start_indices, :2]
        nodes_end = self._mesh.nodes[panel_end_indices, :2]
    
        # Perturbation potential from all source panels
        phi_perturbation = np.zeros(n_panels)
    
        for i in range(n_panels):
            point = centers[i]
        
            for j in range(n_panels):
                if i == j:
                    # Self-influence: use K&P Eq. 10.22a
                    # At panel center: Φ = (σ/4π) * S * ln(S/2)² where S = panel length
                    S = self._mesh.areas[j]
                    self_coeff = S * np.log((S/2)**2)
                else:
                    self_coeff = compute_source_potential_influence(
                        point=point,
                        panel_start=nodes_start[j],
                        panel_end=nodes_end[j]
                    )
            
                phi_perturbation[i] += self._sigma[j] * self_coeff / (4 * np.pi)
    
        # Add freestream potential: Φ_∞ = U_∞*x + W_∞*z
        phi_freestream = (
            self.v_inf_vector[0] * centers[:, 0] + 
            self.v_inf_vector[1] * centers[:, 1]
        )
    
        phi_total = phi_perturbation + phi_freestream
    
        return phi_total
    
    
    def _compute_surface_velocity_from_potential(self) -> NDArray[np.float64]:
        """
        Compute tangential velocity by differentiating potential along surface.
    
        V_t = -∂Φ/∂s where s is arc length along surface
    
        Returns:
            (N,) array of tangential velocities
        """
        phi = self._compute_surface_potential()
        n_panels = len(phi)
    
        # Compute arc length at panel centers
        centers = self._mesh.centers[:, :2]
    
        # Arc length: cumulative distance along surface
        # Assume panels are ordered sequentially around the body
        arc_length = np.zeros(n_panels)
        for i in range(1, n_panels):
            arc_length[i] = arc_length[i-1] + np.linalg.norm(centers[i] - centers[i-1])
    
        # Close the loop: distance from last panel to first
        total_arc = arc_length[-1] + np.linalg.norm(centers[0] - centers[-1])
    
        # Numerical differentiation: central difference with periodic BC
        Vt = np.zeros(n_panels)
    
        for i in range(n_panels):
            # Indices for central difference
            i_plus = (i + 1) % n_panels
            i_minus = (i - 1) % n_panels
        
            # Arc length differences (handle wrap-around)
            if i == 0:
                ds_minus = arc_length[0] + (total_arc - arc_length[-1])  # wrap
            else:
                ds_minus = arc_length[i] - arc_length[i_minus]
        
            if i == n_panels - 1:
                ds_plus = total_arc - arc_length[i]  # wrap to first
            else:
                ds_plus = arc_length[i_plus] - arc_length[i]
        
            ds_total = ds_minus + ds_plus
        
            # Central difference: note the NEGATIVE sign (V_t = -dΦ/ds)
            # Wait - actually for source panels, V_t = dΦ/ds (tangent points in +s direction)
            # The sign depends on convention. Let's use V_t = (Φ_plus - Φ_minus) / ds
            Vt[i] = (phi[i_plus] - phi[i_minus]) / ds_total
    
        return Vt    