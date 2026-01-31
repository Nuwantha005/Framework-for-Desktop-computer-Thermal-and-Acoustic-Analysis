"""
Solver factory with registry pattern.

Enables creation of solver instances based on configuration without
hardcoding specific solver classes throughout the codebase.
"""

from typing import Dict, Type, Tuple, Optional, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from core.geometry.mesh import Mesh
    from core.config.schemas import SolverConfig
    from .panel2d.base import PanelSolver2D


# Registry key: (singularity_type, panel_order, panel_geometry)
RegistryKey = Tuple[str, str, str]


class SolverFactory:
    """
    Factory for creating panel method solvers.
    
    Usage:
        # Register a solver (done in __init__.py)
        SolverFactory.register("source", "constant", "flat", SourcePanelSolver)
        
        # Create solver from config
        solver = SolverFactory.create(config, mesh, v_inf, aoa)
        
        # Or with explicit parameters
        solver = SolverFactory.create_panel_solver(
            singularity="source",
            mesh=mesh,
            v_inf=10.0,
            aoa=5.0
        )
    """
    
    _registry: Dict[RegistryKey, Type["PanelSolver2D"]] = {}
    
    @classmethod
    def register(
        cls,
        singularity_type: str,
        panel_order: str,
        panel_geometry: str,
        solver_class: Type["PanelSolver2D"]
    ) -> None:
        """
        Register a solver class for a specific configuration.
        
        Args:
            singularity_type: "source", "doublet", "vortex", etc.
            panel_order: "constant", "linear", "quadratic"
            panel_geometry: "flat", "curved"
            solver_class: Solver class to instantiate
        """
        key = (singularity_type, panel_order, panel_geometry)
        if key in cls._registry:
            warnings.warn(
                f"Overwriting existing solver registration for {key}",
                UserWarning
            )
        cls._registry[key] = solver_class
    
    @classmethod
    def create(
        cls,
        config: "SolverConfig",
        mesh: "Mesh",
        v_inf: float,
        aoa: float
    ) -> "PanelSolver2D":
        """
        Create a solver from SolverConfig.
        
        Args:
            config: Solver configuration from case.yaml
            mesh: Panel mesh
            v_inf: Freestream velocity magnitude
            aoa: Angle of attack in degrees
        
        Returns:
            Configured solver instance
        """
        return cls.create_panel_solver(
            singularity=config.singularity_type,
            order=config.panel_order,
            geometry=config.panel_geometry,
            mesh=mesh,
            v_inf=v_inf,
            aoa=aoa
        )
    
    @classmethod
    def create_panel_solver(
        cls,
        singularity: str,
        order: str = "constant",
        geometry: str = "flat",
        mesh: "Mesh" = None,
        v_inf: float = 1.0,
        aoa: float = 0.0
    ) -> "PanelSolver2D":
        """
        Create a panel solver with explicit parameters.
        
        Args:
            singularity: Singularity type
            order: Panel order (default: "constant")
            geometry: Panel geometry (default: "flat")
            mesh: Panel mesh
            v_inf: Freestream velocity
            aoa: Angle of attack in degrees
        
        Returns:
            Solver instance
        
        Raises:
            ValueError: If no solver registered for the configuration
        """
        key = (singularity, order, geometry)
        
        if key in cls._registry:
            solver_class = cls._registry[key]
            return solver_class(mesh=mesh, v_inf=v_inf, aoa=aoa)
        
        # Try partial matches with defaults
        fallback_keys = [
            (singularity, "constant", "flat"),  # Try with defaults
            (singularity, order, "flat"),       # Try without curved
        ]
        
        for fallback_key in fallback_keys:
            if fallback_key in cls._registry and fallback_key != key:
                warnings.warn(
                    f"Exact solver for {key} not found. "
                    f"Using fallback: {fallback_key}",
                    UserWarning
                )
                solver_class = cls._registry[fallback_key]
                return solver_class(mesh=mesh, v_inf=v_inf, aoa=aoa)
        
        available = list(cls._registry.keys())
        raise ValueError(
            f"No solver registered for {key}. "
            f"Available configurations: {available}"
        )
    
    @classmethod
    def available(cls) -> Dict[RegistryKey, str]:
        """
        List available solver configurations.
        
        Returns:
            Dict mapping (singularity, order, geometry) to solver class name
        """
        return {k: v.__name__ for k, v in cls._registry.items()}
    
    @classmethod
    def is_registered(cls, singularity: str, order: str = "constant", geometry: str = "flat") -> bool:
        """Check if a solver configuration is registered."""
        return (singularity, order, geometry) in cls._registry
