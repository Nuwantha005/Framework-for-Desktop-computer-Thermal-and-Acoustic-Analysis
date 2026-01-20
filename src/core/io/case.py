"""
Case class - unified container for all case data.

Provides clean access to:
- Scene (geometry + components)
- Flow conditions
- Visualization settings
- Solver settings
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from ..geometry import Scene, Mesh
from ..config.schemas import SimulationConfig


@dataclass
class Case:
    """
    Unified container for a simulation case.
    
    Provides direct attribute access to commonly used values:
        case.name
        case.scene
        case.mesh  (assembled mesh)
        case.v_inf
        case.aoa
        case.x_range
        case.y_range
        case.resolution
    
    Usage:
        from core.io import CaseLoader
        
        case = CaseLoader.load_case('cases/cylinder_flow')
        print(case.name)
        print(case.x_range, case.y_range)
        solver = Solver(case.mesh, case.v_inf, case.aoa)
    """
    
    scene: Scene
    config: SimulationConfig
    case_dir: Path
    
    # Cached mesh
    _mesh: Optional[Mesh] = None
    
    @property
    def name(self) -> str:
        """Case name."""
        return self.config.name
    
    @property
    def description(self) -> str:
        """Case description."""
        return self.config.description
    
    @property
    def mesh(self) -> Mesh:
        """Assembled mesh (cached)."""
        if self._mesh is None:
            self._mesh = self.scene.assemble()
        return self._mesh
    
    @property
    def num_panels(self) -> int:
        """Total number of panels."""
        return self.mesh.num_panels
    
    @property
    def num_components(self) -> int:
        """Number of components."""
        return self.scene.num_components
    
    # -------------------------------------------------------------------------
    # Flow Conditions
    # -------------------------------------------------------------------------
    
    @property
    def freestream(self) -> NDArray:
        """Freestream velocity vector (3,)."""
        return self.scene.freestream
    
    @property
    def v_inf(self) -> float:
        """Freestream velocity magnitude."""
        return float(np.linalg.norm(self.scene.freestream))
    
    @property
    def aoa(self) -> float:
        """Angle of attack in degrees (from freestream direction)."""
        vx, vy = self.scene.freestream[0], self.scene.freestream[1]
        return float(np.degrees(np.arctan2(vy, vx)))
    
    # -------------------------------------------------------------------------
    # Visualization Settings
    # -------------------------------------------------------------------------
    
    @property
    def x_range(self) -> Tuple[float, float]:
        """Visualization x-domain, with auto-calculation if not specified."""
        viz = self.config.visualization
        if viz.domain and 'x_range' in viz.domain:
            return tuple(viz.domain['x_range'])
        # Auto-calculate from mesh
        return self._auto_x_range()
    
    @property
    def y_range(self) -> Tuple[float, float]:
        """Visualization y-domain, with auto-calculation if not specified."""
        viz = self.config.visualization
        if viz.domain and 'y_range' in viz.domain:
            return tuple(viz.domain['y_range'])
        # Auto-calculate from mesh
        return self._auto_y_range()
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Grid resolution (nx, ny)."""
        return self.config.visualization.get_resolution()
    
    @property
    def show_normals(self) -> bool:
        """Whether to show normals in mesh plots."""
        return self.config.visualization.show_normals
    
    def _auto_x_range(self) -> Tuple[float, float]:
        """Calculate x-range from mesh bounds."""
        x_min = self.mesh.nodes[:, 0].min()
        x_max = self.mesh.nodes[:, 0].max()
        padding = (x_max - x_min) * 0.5
        return (float(x_min - padding), float(x_max + 2 * padding))
    
    def _auto_y_range(self) -> Tuple[float, float]:
        """Calculate y-range from mesh bounds."""
        y_min = self.mesh.nodes[:, 1].min()
        y_max = self.mesh.nodes[:, 1].max()
        padding = (y_max - y_min) * 0.5
        return (float(y_min - padding), float(y_max + padding))
    
    # -------------------------------------------------------------------------
    # Solver Settings
    # -------------------------------------------------------------------------
    
    @property
    def solver_type(self) -> str:
        """Solver type string."""
        return self.config.solver.type
    
    @property
    def solver_tolerance(self) -> float:
        """Solver tolerance."""
        return self.config.solver.tolerance
    
    # -------------------------------------------------------------------------
    # Fluid Properties
    # -------------------------------------------------------------------------
    
    @property
    def density(self) -> float:
        """Fluid density [kg/m³]."""
        return self.config.fluid.density
    
    @property
    def viscosity(self) -> Optional[float]:
        """Dynamic viscosity [Pa·s], if specified."""
        return self.config.fluid.viscosity
    
    @property
    def gravity(self) -> float:
        """Gravitational acceleration [m/s²]."""
        return self.config.fluid.gravity
    
    @property
    def reference_pressure(self) -> float:
        """Reference pressure [Pa]."""
        return self.config.fluid.reference_pressure
    
    def get_fluid_state(self) -> 'FluidState':
        """
        Create FluidState object from case config.
        
        Returns:
            FluidState for post-processing calculations
        """
        # Import here to avoid circular dependency
        from postprocessing.fluid import FluidState, ReferenceCondition, ReferenceType
        
        ref_type = ReferenceType(self.config.fluid.reference_type)
        ref = ReferenceCondition(
            type=ref_type,
            pressure=self.config.fluid.reference_pressure,
            velocity=self.v_inf
        )
        
        return FluidState(
            density=self.config.fluid.density,
            reference=ref,
            gravity=self.config.fluid.gravity,
            viscosity=self.config.fluid.viscosity,
            thermal_conductivity=self.config.fluid.thermal_conductivity,
            specific_heat_cp=self.config.fluid.specific_heat_cp
        )
    
    # -------------------------------------------------------------------------
    # Output Paths
    # -------------------------------------------------------------------------
    
    @property
    def output_dir(self) -> Path:
        """Output directory path (case_dir/out)."""
        return self.case_dir / "out"
    
    def __repr__(self) -> str:
        return (
            f"Case(name='{self.name}', "
            f"panels={self.num_panels}, "
            f"components={self.num_components})"
        )
