"""
2D panel method solver base class.

Extends the base Solver interface with panel method specific properties
and configuration.
"""

from abc import abstractmethod
from typing import Tuple, Literal
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from ..base import Solver
from core.geometry.mesh import Mesh


@dataclass(frozen=True)
class PanelMethodConfig:
    """
    Configuration tuple for a panel method solver.
    
    Captures the three orthogonal dimensions of panel method variation:
    - Singularity type: source, doublet, vortex, or combinations
    - Panel order: constant (0th), linear (1st), quadratic (2nd)
    - Panel geometry: flat panels or curved panels
    
    This tuple serves as the registry key in the solver factory.
    """
    singularity_type: Literal[
        "source", "doublet", "vortex",
        "source_doublet", "source_vortex",
        "doublet_vortex", "source_doublet_vortex"
    ]
    panel_order: Literal["constant", "linear", "quadratic"] = "constant"
    panel_geometry: Literal["flat", "curved"] = "flat"
    
    @property
    def key(self) -> Tuple[str, str, str]:
        """Registry lookup key as (singularity, order, geometry)."""
        return (self.singularity_type, self.panel_order, self.panel_geometry)


class PanelSolver2D(Solver):
    """
    Abstract base class for 2D panel method solvers.
    
    Provides common infrastructure for all 2D panel methods while leaving
    implementation details to subclasses.
    
    Subclasses must implement:
    - config property: Return PanelMethodConfig describing the method
    - _compute_influence_matrices(): Build influence coefficient matrices
    - _solve_linear_system(): Solve for singularity strengths
    - _compute_surface_velocity(): Compute Vt from singularity strengths
    - _velocity_at_points(): Compute velocity at arbitrary points
    
    The base class handles:
    - Initialization with mesh and flow conditions
    - Freestream vector computation
    - Common solve() workflow
    - AoA unit consistency (always degrees at API level)
    """
    
    def __init__(self, mesh: Mesh, v_inf: float, aoa: float):
        """
        Initialize 2D panel solver.
        
        Args:
            mesh: 2D panel mesh
            v_inf: Freestream velocity magnitude
            aoa: Angle of attack in DEGREES
        """
        if mesh.dimension != 2:
            raise ValueError("PanelSolver2D requires a 2D mesh")
        
        self._mesh = mesh
        self._v_inf = v_inf
        self._aoa_deg = aoa
        self._aoa_rad = np.radians(aoa)
        
        # Freestream velocity vector
        self._v_inf_vec = np.array([
            v_inf * np.cos(self._aoa_rad),
            v_inf * np.sin(self._aoa_rad),
            0.0  # z-component (always 0 for 2D)
        ])
        
        # Results (populated by solve())
        self._surface_velocity: NDArray[np.float64] = None
        self._solved = False
    
    # --- Public properties ---
    
    @property
    def mesh(self) -> Mesh:
        """Access to the panel mesh."""
        return self._mesh
    
    @property
    def v_inf(self) -> float:
        """Freestream velocity magnitude."""
        return self._v_inf
    
    @property
    def aoa_deg(self) -> float:
        """Angle of attack in degrees."""
        return self._aoa_deg
    
    @property
    def aoa_rad(self) -> float:
        """Angle of attack in radians."""
        return self._aoa_rad
    
    @property
    def v_inf_vector(self) -> NDArray[np.float64]:
        """Freestream velocity vector (vx, vy, 0)."""
        return self._v_inf_vec
    
    @property
    @abstractmethod
    def config(self) -> PanelMethodConfig:
        """Return configuration describing this solver's method."""
        pass
    
    # --- Solver interface implementation ---
    
    def solve(self) -> None:
        """
        Execute the panel method solver.
        
        Standard workflow:
        1. Compute influence coefficient matrices
        2. Assemble and solve linear system for singularity strengths
        3. Compute surface velocity from strengths
        4. Store results in mesh.cell_data for backward compatibility
        """
        # Step 1: Build influence matrices
        influence_matrices = self._compute_influence_matrices()
        
        # Step 2: Solve for singularity strengths
        strengths = self._solve_linear_system(influence_matrices)
        
        # Step 3: Compute surface velocity
        self._surface_velocity = self._compute_surface_velocity(
            influence_matrices, strengths
        )
        
        # Store in mesh for backward compatibility with existing code
        self._mesh.cell_data['Vt'] = self._surface_velocity
        
        # Mark as solved
        self._solved = True
    
    @property
    def surface_velocity(self) -> NDArray[np.float64]:
        """
        Velocity at panel centers.
        
        For 2D: tangential velocity in surface-aligned coordinates.
        
        Returns:
            (N, 3) array with columns [Vt, Vn, 0] where Vn=0 for BC satisfaction
        """
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        
        # Return as (N, 3) with Vt in first column, Vn=0 (BC), Vz=0 (2D)
        n = len(self._surface_velocity)
        result = np.zeros((n, 3), dtype=np.float64)
        result[:, 0] = self._surface_velocity  # Vt
        # result[:, 1] = 0  # Vn (already zero - BC satisfied)
        # result[:, 2] = 0  # Vz (already zero - 2D)
        return result
    
    def velocity_at(
        self,
        points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute velocity at arbitrary field points.
        
        Args:
            points: (M, 2) or (M, 3) array of coordinates
        
        Returns:
            (M, 3) array with columns [Vx, Vy, Vz] where Vz=0 for 2D
        """
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        
        # Ensure points are 2D array
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] == 2:
            # Pad with zeros for z-coordinate
            points = np.column_stack([points, np.zeros(len(points))])
        
        # Get Vx, Vy from subclass implementation
        Vx, Vy = self._velocity_at_points(points)
        
        # Return as (N, 3) array
        result = np.zeros((len(points), 3), dtype=np.float64)
        result[:, 0] = Vx
        result[:, 1] = Vy
        # result[:, 2] = 0  # Vz (already zero - 2D)
        return result
    
    # --- Abstract methods for subclasses ---
    
    @abstractmethod
    def _compute_influence_matrices(self) -> dict:
        """
        Compute influence coefficient matrices.
        
        Returns:
            Dict with keys like "I" (normal influence), "J" (tangential influence).
            Specific keys depend on singularity type.
        """
        pass
    
    @abstractmethod
    def _solve_linear_system(self, influence_matrices: dict) -> dict:
        """
        Solve for singularity strengths.
        
        Args:
            influence_matrices: From _compute_influence_matrices()
        
        Returns:
            Dict mapping singularity type to strength array.
            e.g., {"source": sigma_array}
        """
        pass
    
    @abstractmethod
    def _compute_surface_velocity(
        self,
        influence_matrices: dict,
        strengths: dict
    ) -> NDArray[np.float64]:
        """
        Compute tangential velocity at panel centers from singularity strengths.
        
        Args:
            influence_matrices: Influence coefficient matrices
            strengths: Singularity strength arrays
        
        Returns:
            (N,) array of surface velocity magnitudes
        """
        pass
    
    @abstractmethod
    def _velocity_at_points(
        self,
        points: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute velocity induced by singularities at arbitrary points.
        
        This is the solver-specific velocity calculation that replaces
        the old VelocityField2D._compute_point_velocity().
        
        Args:
            points: (M, 3) array of (x, y, z) coordinates
        
        Returns:
            (Vx, Vy) tuple of (M,) arrays
        """
        pass
    
    # --- Backward compatibility properties ---
    
    @property
    def Vt(self) -> NDArray[np.float64]:
        """Surface velocity (backward compatibility)."""
        return self.surface_velocity
    
    @property
    def Cp(self) -> NDArray[np.float64]:
        """
        Pressure coefficient (computed via Bernoulli).
        
        Note: This is kept for backward compatibility but Cp should
        ideally be computed externally using the PressureProcessor.
        """
        if not self._solved:
            raise RuntimeError("Solver not executed. Call solve() first.")
        return 1.0 - (self._surface_velocity / self._v_inf) ** 2
