"""
Thermal solver factory for creating solvers from case configuration.

Provides a unified interface for instantiating thermal solvers based on
the solver type specified in the case YAML file.
"""

from typing import Optional, Union, TYPE_CHECKING

from .base import ThermalBLInput, ThermalSolver, ThermalSolverConfig
from .reynolds_analogy import ReynoldsAnalogyThermal

if TYPE_CHECKING:
    from core.io.case import Case
    from solvers.boundary_layer.runner import BoundaryLayerPathResult
    from solvers.boundary_layer.field import BLFieldData


# Registry of available thermal solvers
_SOLVER_REGISTRY = {
    "reynolds_analogy": ReynoldsAnalogyThermal,
    "bdim": "bdim",  # Special marker - requires different creation path
}

# Try to import BDIM components
try:
    from .bdim.solver import BDIMThermalSolver, BDIMInput, BDIMConfig
    from .bdim.extraction import extract_bdim_input_from_bl_field
    _BDIM_AVAILABLE = True
except ImportError:
    _BDIM_AVAILABLE = False


def create_thermal_solver(
    solver_type: str,
    bl_input: ThermalBLInput,
    config: ThermalSolverConfig,
) -> ThermalSolver:
    """
    Create a thermal solver by type name.
    
    Args:
        solver_type: One of "reynolds_analogy" or "bdim"
        bl_input: Input data from viscous BL solver
        config: Thermal solver configuration
    
    Returns:
        Configured ThermalSolver instance
    
    Raises:
        ValueError: If solver_type is unknown or not available
    """
    solver_type = solver_type.lower()
    
    if solver_type == "reynolds_analogy":
        return ReynoldsAnalogyThermal(bl_input, config)
    
    elif solver_type == "bdim":
        raise NotImplementedError(
            "BDIM solver requires domain mesh data from BL field reconstruction. "
            "Use create_bdim_solver() with BL path and field data, or use "
            "'reynolds_analogy' for surface-only thermal analysis."
        )
    
    else:
        available = list(_SOLVER_REGISTRY.keys())
        raise ValueError(
            f"Unknown thermal solver type '{solver_type}'. "
            f"Available: {available}"
        )


def create_bdim_solver(
    bl_path: "BoundaryLayerPathResult",
    bl_field: "BLFieldData",
    config: ThermalSolverConfig,
) -> "BDIMThermalSolver":
    """
    Create BDIM thermal solver from BL path and field data.
    
    This is the proper way to instantiate the BDIM solver, as it requires
    domain mesh data from the BL field reconstruction.
    
    Args:
        bl_path: BoundaryLayerPathResult from BoundaryLayerRunner
        bl_field: BLFieldData from reconstruct_bl_field()
        config: Thermal solver configuration (T_inf, Pr, k, q_wall/T_wall, etc.)
    
    Returns:
        Configured BDIMThermalSolver ready to solve
    
    Raises:
        ImportError: If BDIM module not available
        ValueError: If field data doesn't match path data
    
    Example::
    
        from solvers.thermal import create_bdim_solver, ThermalSolverConfig
        from solvers.boundary_layer.field import reconstruct_bl_field
        
        # Run BL solver with reconstruction
        bl = runner.run(profiles=["thwaites"], reconstruct=True)
        field = bl.upper.fields["thwaites"]
        
        # Create and run BDIM solver
        config = ThermalSolverConfig(T_inf=300.0, q_wall=1000.0, Pr=0.71, k=0.026)
        solver = create_bdim_solver(bl.upper, field, config)
        result = solver.solve()
    """
    if not _BDIM_AVAILABLE:
        raise ImportError(
            "BDIM solver not available. Check that solvers.thermal.bdim "
            "module is properly installed."
        )
    
    # Extract BDIM input from BL data (handles coordinate transformation)
    bdim_input = extract_bdim_input_from_bl_field(bl_path, bl_field)
    
    # Convert ThermalSolverConfig to BDIMConfig
    # BDIMConfig uses slightly different naming (mu instead of derived from Pr)
    mu = (config.cp * config.Pr) / config.k if config.Pr else 1.81e-5
    
    bdim_config = BDIMConfig(
        T_inf=config.T_inf,
        rho=config.rho,
        mu=mu,
        k=config.k,
        cp=config.cp,
        q_wall=config.q_wall,
        T_wall=config.T_wall,
    )
    
    # Create solver
    return BDIMThermalSolver(
        bdim_input=bdim_input,
        config=bdim_config,
    )


class ThermalSolverFactory:
    """
    Factory for creating thermal solvers from case configuration.
    
    Reads thermal settings from case.yaml and creates appropriately
    configured solver instances.
    
    Example::
    
        from core.io import CaseLoader
        from solvers.thermal import ThermalSolverFactory
        from solvers.thermal.base import extract_thermal_input
        
        case = CaseLoader.load_case("cases/cylinder_flow")
        
        # Get thermal input from BL result
        thermal_input = extract_thermal_input(bl_result.upper, "thwaites")
        
        # Create solver from case config
        solver = ThermalSolverFactory.create_from_case(case, thermal_input)
        result = solver.solve()
    """
    
    @staticmethod
    def create_from_case(
        case: "Case",
        bl_input: ThermalBLInput,
        component_name: Optional[str] = None,
        q_wall_override: Optional[float] = None,
        T_wall_override: Optional[float] = None,
    ) -> ThermalSolver:
        """
        Create thermal solver from case configuration.
        
        Args:
            case: Loaded Case object with thermal config
            bl_input: Input data from viscous BL solver
            component_name: Component to get heat_flux BC from (default: first component)
            q_wall_override: Override heat flux BC from case file
            T_wall_override: Override temperature BC (use instead of heat flux)
        
        Returns:
            Configured ThermalSolver instance
        
        Raises:
            ValueError: If thermal section missing or no valid BC provided
        """
        # Get thermal config
        thermal_cfg = case.config.thermal
        fluid_cfg = case.config.fluid
        
        # Get solver type
        solver_type = thermal_cfg.solver
        
        # Get freestream temperature
        T_inf = fluid_cfg.freestream_temperature
        if T_inf is None:
            T_inf = 300.0  # Default room temperature
        
        # Get fluid properties (with air defaults)
        k = fluid_cfg.thermal_conductivity if fluid_cfg.thermal_conductivity else 0.026
        cp = fluid_cfg.specific_heat_cp if fluid_cfg.specific_heat_cp else 1005.0
        rho = fluid_cfg.density
        mu = fluid_cfg.viscosity if fluid_cfg.viscosity else 1.81e-5
        
        # Compute Prandtl number
        Pr = (cp * mu) / k
        
        # Get boundary condition
        q_wall = q_wall_override
        T_wall = T_wall_override
        
        if q_wall is None and T_wall is None:
            # Try to get from component BC
            q_wall = ThermalSolverFactory._get_component_heat_flux(
                case, component_name
            )
        
        if q_wall is None and T_wall is None:
            raise ValueError(
                "No thermal BC specified. Provide heat_flux in component "
                "boundary_condition, or pass q_wall_override/T_wall_override."
            )
        
        # Build config
        config = ThermalSolverConfig(
            T_inf=T_inf,
            Pr=Pr,
            k=k,
            rho=rho,
            cp=cp,
            q_wall=q_wall,
            T_wall=T_wall,
        )
        
        return create_thermal_solver(solver_type, bl_input, config)
    
    @staticmethod
    def _get_component_heat_flux(
        case: "Case",
        component_name: Optional[str] = None,
    ) -> Optional[float]:
        """
        Get heat flux BC from component configuration.
        
        Args:
            case: Loaded Case object
            component_name: Component name, or None for first component
        
        Returns:
            Heat flux value [W/m²] or None if not specified
        """
        components = case.scene.components
        
        if not components:
            return None
        
        if component_name is None:
            # Use first component
            comp = components[0]
        else:
            # Find named component
            comp = None
            for c in components:
                if c.name == component_name:
                    comp = c
                    break
            if comp is None:
                raise ValueError(f"Component '{component_name}' not found")
        
        return comp.bc_heat_flux
    
    @staticmethod
    def available_solvers() -> list:
        """Return list of available thermal solver types."""
        return list(_SOLVER_REGISTRY.keys())
