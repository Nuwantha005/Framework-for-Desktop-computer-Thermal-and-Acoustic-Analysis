"""
Thermal boundary layer solvers.

This module provides thermal BL solvers that compute wall temperature,
heat transfer coefficients, and Nusselt number from viscous BL data.

Two solver types are available:

1. **Reynolds Analogy** (``reynolds_analogy``): Fast baseline using Chilton-Colburn
   analogy. Derives heat transfer directly from skin friction. Suitable for
   attached boundary layers with moderate pressure gradients.

2. **BDIM** (``bdim``): Full Boundary-Domain Integral Method from Gao et al. (2013).
   More accurate for complex geometries and separated regions, but requires
   domain mesh and velocity gradient data.

Usage
-----
The recommended workflow is:

1. Run viscous BL solver to get BoundaryLayerPathResult
2. Extract thermal input: ``thermal_input = extract_thermal_input(path_result, profile)``
3. Create solver config with BCs: ``config = ThermalSolverConfig(T_inf=300, q_wall=1000)``
4. Create and run solver: ``result = create_thermal_solver("reynolds_analogy", input, config).solve()``

Or use the factory directly with case configuration::

    from solvers.thermal import ThermalSolverFactory
    
    solver = ThermalSolverFactory.create_from_case(case, thermal_input)
    result = solver.solve()

Example
-------
::

    from solvers.thermal import (
        ReynoldsAnalogyThermal,
        ThermalSolverConfig,
        extract_thermal_input,
        create_thermal_solver,
    )
    
    # From viscous BL result
    thermal_input = extract_thermal_input(bl_case_result.upper, "thwaites")
    
    # Configure with heat flux BC
    config = ThermalSolverConfig(T_inf=300.0, q_wall=1000.0, Pr=0.71, k=0.026)
    
    # Create solver (can switch type easily)
    solver = create_thermal_solver("reynolds_analogy", thermal_input, config)
    result = solver.solve()
    
    print(f"Total heat rate: {result.total_heat_rate:.2f} W/m")
"""

from .base import (
    ThermalBLInput,
    ThermalResult,
    ThermalSolver,
    ThermalSolverConfig,
    extract_thermal_input,
)
from .reynolds_analogy import ReynoldsAnalogyThermal
from .factory import ThermalSolverFactory, create_thermal_solver

# BDIM solver import (may not be fully functional yet)
try:
    from .bdim.solver import BDIMThermalSolver
    _BDIM_AVAILABLE = True
except ImportError:
    BDIMThermalSolver = None
    _BDIM_AVAILABLE = False

__all__ = [
    # Input/output data structures
    "ThermalBLInput",
    "ThermalResult",
    "ThermalSolverConfig",
    "extract_thermal_input",
    # Base class
    "ThermalSolver",
    # Concrete solvers
    "ReynoldsAnalogyThermal",
    "BDIMThermalSolver",
    # Factory
    "ThermalSolverFactory",
    "create_thermal_solver",
]
