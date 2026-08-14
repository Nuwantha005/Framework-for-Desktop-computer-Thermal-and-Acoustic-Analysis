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


# Registry key: (dimension, singularity_type, panel_order, panel_geometry)
RegistryKey = Tuple[int, str, str, str]


class SolverFactory:
    """
    Factory for creating panel method solvers.
    
    Usage:
        # Register a solver (done in __init__.py)
        SolverFactory.register(2, "source", "constant", "flat", SourcePanelSolver)
        
        # Create solver from config
        solver = SolverFactory.create(config, mesh, v_inf, aoa)
        
        # Or with explicit parameters
        solver = SolverFactory.create_panel_solver(
            dimension=2,
            singularity="source",
            mesh=mesh,
            v_inf=10.0,
            aoa=5.0
        )
    """
    
    _registry: Dict[RegistryKey, Type] = {}
    
    @classmethod
    def register(
        cls,
        dimension: int,
        singularity_type: str,
        panel_order: str,
        panel_geometry: str,
        solver_class: Type
    ) -> None:
        """
        Register a solver class for a specific configuration.
        
        Args:
            dimension: 2 or 3 (mesh dimension)
            singularity_type: "source", "doublet", "vortex", etc.
            panel_order: "constant", "linear", "quadratic"
            panel_geometry: "flat", "curved"
            solver_class: Solver class to instantiate
        """
        key = (dimension, singularity_type, panel_order, panel_geometry)
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
    ):
        """
        Create a solver from SolverConfig.
        
        Args:
            config: Solver configuration from case.yaml
            mesh: Panel mesh (2D or 3D)
            v_inf: Freestream velocity magnitude
            aoa: Angle of attack in degrees
        
        Returns:
            Configured solver instance
        """
        return cls.create_panel_solver(
            dimension=mesh.dimension,
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
        dimension: int,
        singularity: str,
        order: str = "constant",
        geometry: str = "flat",
        mesh: "Mesh" = None,
        v_inf: float = 1.0,
        aoa: float = 0.0
    ):
        """
        Create a panel solver with explicit parameters.
        
        Args:
            dimension: 2 or 3
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
        key = (dimension, singularity, order, geometry)
        
        if key in cls._registry:
            solver_class = cls._registry[key]
            if dimension == 3:
                import numpy as np
                v_arr = np.asarray(v_inf, dtype=np.float64)
                if v_arr.shape == (3,):
                    v_inf_vec = v_arr
                else:
                    speed = float(v_arr)
                    v_inf_vec = np.array([
                        speed * np.cos(np.deg2rad(aoa)),
                        speed * np.sin(np.deg2rad(aoa)),
                        0.0
                    ])
                return solver_class(mesh=mesh, v_inf=v_inf_vec)
            else:
                return solver_class(mesh=mesh, v_inf=v_inf, aoa=aoa)
        
        # Try partial matches with defaults
        fallback_keys = [
            (dimension, singularity, "constant", "flat"),  # Try with defaults
            (dimension, singularity, order, "flat"),       # Try without curved
        ]
        
        for fallback_key in fallback_keys:
            if fallback_key in cls._registry and fallback_key != key:
                warnings.warn(
                    f"Exact solver for {key} not found. "
                    f"Using fallback: {fallback_key}",
                    UserWarning
                )
                solver_class = cls._registry[fallback_key]
                if dimension == 3:
                    import numpy as np
                    v_arr = np.asarray(v_inf, dtype=np.float64)
                    if v_arr.shape == (3,):
                        v_inf_vec = v_arr
                    else:
                        speed = float(v_arr)
                        v_inf_vec = np.array([
                            speed * np.cos(np.deg2rad(aoa)),
                            speed * np.sin(np.deg2rad(aoa)),
                            0.0
                        ])
                    return solver_class(mesh=mesh, v_inf=v_inf_vec)
                else:
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
            Dict mapping (dimension, singularity, order, geometry) to solver class name
        """
        return {k: v.__name__ for k, v in cls._registry.items()}
    
    @classmethod
    def is_registered(cls, dimension: int, singularity: str, order: str = "constant", geometry: str = "flat") -> bool:
        """Check if a solver configuration is registered."""
        return (dimension, singularity, order, geometry) in cls._registry
