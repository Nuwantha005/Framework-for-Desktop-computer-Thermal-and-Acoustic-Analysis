import numpy as np
from numpy.typing import NDArray
from typing import Dict

from core.geometry.mesh import Mesh
from .base import PanelSolver2D, PanelMethodConfig
from .influences.linear_vortex import (
    compute_linear_vortex_influence_matrices,
    compute_linear_vortex_velocity_field
)

class LinearVortexPanelSolver(PanelSolver2D):
    """
    2D linear-strength vortex panel method solver.
    Adapted for non-lifting bluff bodies using Zero Net Circulation closure.
    """
    
    @property
    def config(self) -> PanelMethodConfig:
        return PanelMethodConfig(
            singularity_type="vortex",
            panel_order="linear",
            panel_geometry="flat"
        )
        
    @property
    def gamma(self) -> NDArray[np.float64]:
        """Node vortex strengths."""
        if self._gamma is None:
            raise RuntimeError("Solver not yet run. Call solve() first.")
        return self._gamma

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
        return 1.0 - (self.Vt / self._v_inf) ** 2

        

    def __init__(self, mesh: Mesh, v_inf: float = 1.0, aoa: float = 0.0):
        super().__init__(mesh, v_inf, aoa)
        self._gamma = None
        
    def _compute_influence_matrices(self) -> Dict[str, NDArray]:
        I, J = compute_linear_vortex_influence_matrices(self._mesh)
        return {"I": I, "J": J}
        
    def _solve_linear_system(self, influence_matrices: Dict[str, NDArray]) -> Dict[str, NDArray]:
        I = influence_matrices["I"]
        
        # Original N equations: Normal velocity = 0
        v_inf_2d = self.v_inf_vector[:2]
        normals = self._mesh.normals[:, :2]
        b_panels = -np.dot(normals, v_inf_2d)
        
        # To handle rank-deficiency of Neumann pure-vortex problem, we add 
        # a Zero Circulation constraint for each independent body structure 
        # (each component).
        unique_components = np.unique(self._mesh.component_ids)
        num_components = len(unique_components)
        
        num_panels = self._mesh.num_panels
        num_nodes = self._mesh.num_nodes
        
        A_aug = np.zeros((num_panels + num_components, num_nodes))
        b_aug = np.zeros(num_panels + num_components)
        
        # Populate N panel center equations
        A_aug[:num_panels, :] = I
        b_aug[:num_panels] = b_panels
        
        # Populate 1 circulation constraint per component
        for idx, comp_id in enumerate(unique_components):
            row_idx = num_panels + idx
            # Find panels belonging to this component
            comp_panels = np.where(self._mesh.component_ids == comp_id)[0]
            
            # The total circulation for component is Sum((gamma_a + gamma_b)/2 * S_j)
            for j in comp_panels:
                n1_idx = self._mesh.panels[j, 0]
                n2_idx = self._mesh.panels[j, 1]
                S_j = self._mesh.areas[j]
                
                A_aug[row_idx, n1_idx] += 0.5 * S_j
                A_aug[row_idx, n2_idx] += 0.5 * S_j
                
            b_aug[row_idx] = 0.0 # Zero net circulation
            
            # Weigh the equation to balance the matrix values (I matrix is ~ 0.5)
            avg_area = np.mean(self._mesh.areas[comp_panels])
            A_aug[row_idx, :] /= avg_area
            
        try:
            # Overdetermined system perfectly solved by least squares
            gamma, residuals, rank, s = np.linalg.lstsq(A_aug, b_aug, rcond=None)
        except np.linalg.LinAlgError:
            raise RuntimeError("Failed to solve linear system for vortex strengths.")
            
        self._gamma = gamma
        self._mesh.cell_data['vortex_strength'] = gamma
        
        return {"vortex": gamma}

    def _compute_surface_velocity(self, influence_matrices: Dict[str, NDArray], strengths: Dict[str, NDArray]) -> NDArray:
        J = influence_matrices["J"]
        gamma = strengths["vortex"]
        
        # Induced tangential velocity in local mesh tangent direction
        Vt_induced = J @ gamma
        
        # Add freestream component
        tangents = self._mesh.tangents[:, :2]
        v_inf_2d = self.v_inf_vector[:2]
        Vt_freestream = np.dot(tangents, v_inf_2d)
        
        return Vt_freestream + Vt_induced

    def _velocity_at_points(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._gamma is None:
            raise RuntimeError("Must solve before evaluating velocity field.")
            
        V_induced = compute_linear_vortex_velocity_field(
            points, self._mesh, self._gamma
        )
        
        V_total = np.zeros((points.shape[0], 3))
        V_total[:, :2] = V_induced + self.v_inf_vector[:2]
        return V_total
        
    def _compute_induced_normal_velocity(self) -> NDArray[np.float64]:
        I = self._influence_matrices["I"]
        return I @ self._gamma
        
    def _perform_solver_specific_validation(self):
        # We can check if net circulation is approximately zero
        unique_components = np.unique(self._mesh.component_ids)
        for comp_id in unique_components:
            comp_panels = np.where(self._mesh.component_ids == comp_id)[0]
            circ = 0.0
            for j in comp_panels:
                n1_idx = self._mesh.panels[j, 0]
                n2_idx = self._mesh.panels[j, 1]
                S_j = self._mesh.areas[j]
                circ += 0.5 * S_j * (self._gamma[n1_idx] + self._gamma[n2_idx])
                
            if abs(circ) > 1e-5:
                # Add logging warning or print in dev mode
                print(f"Warning: Component {comp_id} has non-zero circulation {circ:.2e}")
