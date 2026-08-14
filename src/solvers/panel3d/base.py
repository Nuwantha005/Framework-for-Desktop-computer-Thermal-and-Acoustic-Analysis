"""
3D panel method solver base class.

Extends the base Solver interface with 3D panel method specific properties.
"""

from abc import abstractmethod
from typing import Tuple, Optional
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from ..base import Solver
from core.geometry.mesh3d import Mesh3D


class PanelSolver3D(Solver):
    """
    Abstract base class for 3D panel method solvers.
    
    Provides common infrastructure for 3D panel methods. The key difference
    from 2D solvers is that freestream is specified as a 3D vector rather
    than magnitude + angle of attack.
    
    Subclasses must implement:
    - _compute_influence_matrix(): Build normal velocity influence matrix
    - _solve_linear_system(): Solve for singularity strengths
    - _compute_surface_velocity(): Compute velocity at panel centers
    - _velocity_at_points(): Compute velocity at arbitrary points
    """
    
    def __init__(
        self,
        mesh: Mesh3D,
        v_inf: NDArray[np.float64],
    ):
        """
        Initialize 3D panel solver.
        
        Args:
            mesh: 3D surface mesh with quad panels
            v_inf: Freestream velocity vector (3,) - [Vx, Vy, Vz]
        """
        if mesh.dimension != 3:
            raise ValueError("PanelSolver3D requires a 3D mesh")
        
        self._mesh = mesh
        self._v_inf = np.asarray(v_inf, dtype=np.float64).flatten()
        
        if len(self._v_inf) != 3:
            raise ValueError(f"v_inf must be 3D vector, got shape {self._v_inf.shape}")
        
        # Ensure mesh geometry is computed
        if mesh.centers is None:
            mesh.compute_geometry()
        
        # Results (populated by solve())
        self._surface_velocity: NDArray[np.float64] = None
        self._sigma: NDArray[np.float64] = None  # Source strengths
        self._solved = False
    
    # --- Public properties ---
    
    @property
    def mesh(self) -> Mesh3D:
        """Access to the panel mesh."""
        return self._mesh
    
    @property
    def v_inf(self) -> NDArray[np.float64]:
        """Freestream velocity vector (3,)."""
        return self._v_inf
    
    @property
    def v_inf_magnitude(self) -> float:
        """Freestream velocity magnitude."""
        return float(np.linalg.norm(self._v_inf))
    
    @property
    def is_solved(self) -> bool:
        """Whether the solver has been executed."""
        return self._solved
    
    # --- Solver interface implementation ---
    
    def solve(self, normal_velocity_disturbance: Optional[NDArray[np.float64]] = None) -> None:
        """
        Execute the panel method solver.
        
        Standard workflow:
        1. Compute influence coefficient matrix (normal velocity influence)
        2. Assemble and solve linear system for source strengths
        3. Compute surface velocity from strengths
        4. Store results in mesh.cell_data

        Args:
            normal_velocity_disturbance: Optional known external normal velocity
                at body panel centers. Used by coupled models such as ADM.
        """
        if normal_velocity_disturbance is not None:
            disturbance = np.asarray(normal_velocity_disturbance, dtype=np.float64)
            if disturbance.shape != (self._mesh.num_panels,):
                raise ValueError(
                    "normal_velocity_disturbance must have shape "
                    f"({self._mesh.num_panels},), got {disturbance.shape}"
                )
            self._external_normal_velocity = disturbance
        else:
            self._external_normal_velocity = None

        # Step 1: Build influence matrix
        influence_matrix = self._compute_influence_matrix()
        
        # Store for later use
        self._influence_matrix = influence_matrix
        
        # Step 2: Solve for singularity strengths
        strengths = self._solve_linear_system(influence_matrix)
        
        # Step 3: Compute surface velocity
        self._surface_velocity = self._compute_surface_velocity(strengths)
        
        # Mark as solved (before storing Cp which depends on is_solved)
        self._solved = True
        
        # Store in mesh for visualization/export
        self._mesh.cell_data['sigma'] = self._sigma
        self._mesh.cell_data['Vt'] = np.linalg.norm(self._surface_velocity, axis=1)
        self._mesh.cell_data['Cp'] = self.Cp
    
    @property
    def surface_velocity(self) -> NDArray[np.float64]:
        """
        Velocity at panel centers.
        
        Returns:
            (N, 3) array with [Vx, Vy, Vz] at each panel center
        """
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self._surface_velocity
    
    @property
    def sigma(self) -> NDArray[np.float64]:
        """
        Source strengths for each panel.
        
        Returns:
            (N,) array of source strengths
        """
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return self._sigma
    
    @property
    def Cp(self) -> NDArray[np.float64]:
        """
        Pressure coefficient at panel centers via Bernoulli.
        
        Cp = 1 - (V/V_inf)²
        
        Returns:
            (N,) array of pressure coefficients
        """
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        
        V_mag = np.linalg.norm(self._surface_velocity, axis=1)
        V_inf_mag = self.v_inf_magnitude
        if V_inf_mag <= 1e-14:
            return np.full(self._mesh.num_panels, np.nan, dtype=np.float64)
        
        return 1.0 - (V_mag / V_inf_mag) ** 2
    
    def velocity_at(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute velocity at arbitrary field points.
        
        Args:
            points: (M, 3) array of coordinates
        
        Returns:
            (M, 3) array with [Vx, Vy, Vz] at each point
        """
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        if points.shape[1] != 3:
            raise ValueError(f"Points must be (M, 3), got shape {points.shape}")
        
        return self._velocity_at_points(points)
    
    # --- Abstract methods for subclasses ---
    
    @abstractmethod
    def _compute_influence_matrix(self) -> NDArray[np.float64]:
        """
        Compute influence coefficient matrix for normal velocity.
        
        Element [i, j] = normal velocity at panel i center due to 
        unit-strength singularity on panel j.
        
        Returns:
            (N, N) influence matrix
        """
        pass
    
    @abstractmethod
    def _solve_linear_system(self, influence_matrix: NDArray[np.float64]) -> dict:
        """
        Solve for singularity strengths.
        
        Args:
            influence_matrix: From _compute_influence_matrix()
        
        Returns:
            Dict mapping singularity type to strength array
        """
        pass
    
    @abstractmethod
    def _compute_surface_velocity(self, strengths: dict) -> NDArray[np.float64]:
        """
        Compute velocity at panel centers from singularity strengths.
        
        Args:
            strengths: Singularity strength arrays
        
        Returns:
            (N, 3) array of velocity vectors at panel centers
        """
        pass
    
    @abstractmethod
    def _velocity_at_points(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute velocity at arbitrary field points.
        
        Args:
            points: (M, 3) array of coordinates
        
        Returns:
            (M, 3) array of velocity vectors
        """
        pass
    
    # --- Validation helpers ---
    
    def validate_boundary_condition(self) -> dict:
        """
        Verify that Vn = 0 is satisfied at panel centers.
        
        Returns:
            Dict with Vn statistics
        """
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        
        # Normal velocity at each panel center
        Vn = np.einsum('ij,ij->i', self._surface_velocity, self._mesh.normals)
        
        return {
            "Vn_max_abs": float(np.max(np.abs(Vn))),
            "Vn_rms": float(np.sqrt(np.mean(Vn**2))),
            "Vn_mean": float(np.mean(Vn)),
        }
