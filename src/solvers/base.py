"""
Base solver interface for all panel method and multi-physics solvers.

Defines the minimal interface that all solvers must implement:
- solve() to execute the solver
- surface_velocity property for velocity at panel centers
- velocity_at(points) method for field computation

All velocity outputs use (N, 3) array format with z=0 for 2D cases.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from core.geometry.mesh import Mesh


class Solver(ABC):
    """
    Abstract base class for all solvers.
    
    All panel method solvers must implement this interface to ensure
    consistent downstream usage for visualization, validation, and coupling.
    
    Key outputs:
    - surface_velocity: Velocity at panel centers as (N, 3) array
    - velocity_at(points): Velocity field at arbitrary points as (M, 3) array
    
    All velocity arrays use (N, 3) format for consistency, with z=0 for 2D.
    
    Note: Pressure coefficient (Cp) is computed externally using Bernoulli
    equation, not by the solver itself.
    """
    
    @abstractmethod
    def solve(self) -> None:
        """
        Execute the solver to compute flow solution.
        
        After calling this method, surface_velocity must be available
        and velocity_at() must be callable.
        """
        pass
    
    @property
    @abstractmethod
    def surface_velocity(self) -> NDArray[np.float64]:
        """
        Velocity at panel centers.
        
        For 2D: columns are [Vt, Vn, 0] in surface-aligned coordinates.
        For 3D: columns are [Vx, Vy, Vz] in global coordinates.
        
        Returns:
            (N, 3) array where N is number of panels
        
        Raises:
            RuntimeError: If solver has not been executed yet
        """
        pass
    
    @abstractmethod
    def velocity_at(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute velocity at arbitrary field points.
        
        This method encapsulates the solver-specific velocity calculation.
        Each solver type (source, doublet, vortex) implements this differently
        based on its singularity distribution.
        
        Args:
            points: (M, 2) or (M, 3) array of coordinates
        
        Returns:
            (M, 3) array with columns [Vx, Vy, Vz] in global coordinates.
            For 2D cases, Vz=0.
        
        Raises:
            RuntimeError: If solver has not been executed yet
        
        Note:
            This replaces the VelocityField2D class which was tightly
            coupled to source panels. Each solver now handles its own
            velocity field calculation.
        """
        pass
    
    @property
    def is_solved(self) -> bool:
        """Check if solver has been executed."""
        return hasattr(self, '_solved') and self._solved
    
    @property
    @abstractmethod
    def mesh(self) -> "Mesh":
        """Access to the panel mesh."""
        pass
