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
    
    For parametric cases, provides mesh level control:
        case.mesh_level_index  (current level)
        case.mesh_level  (resolution tuple, e.g., [8, 8])
        case.num_mesh_levels  (total levels available)
        case.reload_at_level(index)  (load different resolution)
    
    Usage:
        from core.io import CaseLoader
        
        case = CaseLoader.load_case('cases/cylinder_flow')
        print(case.name)
        print(case.x_range, case.y_range)
        solver = Solver(case.mesh, case.v_inf, case.aoa)
        
        # For parametric cases:
        case_fine = case.reload_at_level(-1)  # Load finest level
    """
    
    scene: Scene
    config: SimulationConfig
    case_dir: Path
    mesh_level_index: int = 0
    
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
    # Mesh Level Management (for parametric cases)
    # -------------------------------------------------------------------------
    
    @property
    def num_mesh_levels(self) -> int:
        """
        Number of mesh levels available (0 for non-parametric cases).
        
        Returns the number of levels from the first parametric component.
        If components have different numbers of levels, this returns the first one found.
        """
        for comp in self.config.components:
            if comp.mesh_levels is not None:
                return len(comp.mesh_levels)
        return 0
    
    @property
    def mesh_level(self) -> Optional[dict[str, list[int]]]:
        """
        Current mesh resolution for each parametric component.
        
        Returns:
            Dictionary mapping component names to their resolution tuples,
            e.g., {"square": [8, 8], "cylinder": [32]}
            Returns None if no parametric components.
        """
        levels = {}
        for comp in self.config.components:
            if comp.mesh_levels is not None and len(comp.mesh_levels) > 0:
                # Handle negative indexing
                idx = self.mesh_level_index
                if idx < 0:
                    idx = len(comp.mesh_levels) + idx
                if 0 <= idx < len(comp.mesh_levels):
                    levels[comp.name] = comp.mesh_levels[idx]
        return levels if levels else None
    
    def reload_at_level(self, level_index: int) -> Case:
        """
        Reload case at different mesh level.
        
        Args:
            level_index: Index into per-component mesh_levels (use -1 for finest level)
        
        Returns:
            New Case object at specified mesh level
        
        Raises:
            ValueError: If case doesn't use parametric geometry
            IndexError: If level_index out of range for any component
        
        Example:
            >>> case_coarse = CaseLoader.load_case('cases/single_square', mesh_level_index=0)
            >>> case_fine = case_coarse.reload_at_level(-1)  # Finest level
        
        Note:
            The level_index applies to all parametric components.
            All components should have mesh_levels defined with the same length.
        """
        # Check if any component uses parametric geometry
        has_parametric = any(comp.mesh_levels is not None for comp in self.config.components)
        if not has_parametric:
            raise ValueError(f"Case '{self.name}' does not use parametric geometry (no mesh_levels defined)")
        
        # Import here to avoid circular dependency
        from .case_loader import CaseLoader
        return CaseLoader.load_case(self.case_dir, mesh_level_index=level_index)
    
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
    
    def create_solver(self, solver_type: Optional[str] = None) -> "Solver":
        """
        Create solver instance from case configuration.

        Args:
            solver_type: Optional solver type override. Accepts legacy format
                strings such as "constant_source" or "linear_source", or short
                aliases like "constant" / "linear".  When *None*, the solver
                type from ``case.yaml`` is used.

        Returns:
            Solver instance configured from case settings (mesh, v_inf, aoa).

        Example:
            >>> case = CaseLoader.load_case('cases/cylinder_flow')
            >>> solver = case.create_solver()
            >>> solver.solve()

            >>> # Override solver without editing case.yaml
            >>> solver_lin = case.create_solver(solver_type="linear_source")
        """
        from solvers.factory import SolverFactory

        if (
            solver_type is None
            and self.mesh.dimension == 3
            and self.config.actuator_disks
        ):
            from solvers.actuator import ActuatorDiskCoupledSolver3D

            return ActuatorDiskCoupledSolver3D.from_case(self)

        if solver_type is not None:
            from core.config.schemas import SolverConfig
            override = SolverConfig(
                type=solver_type,
                tolerance=self.config.solver.tolerance,
                max_iterations=self.config.solver.max_iterations,
            )
            return SolverFactory.create(
                config=override,
                mesh=self.mesh,
                v_inf=self.freestream if self.mesh.dimension == 3 else self.v_inf,
                aoa=self.aoa,
            )

        return SolverFactory.create(
            config=self.config.solver,
            mesh=self.mesh,
            v_inf=self.freestream if self.mesh.dimension == 3 else self.v_inf,
            aoa=self.aoa,
        )
    
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
