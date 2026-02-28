from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

@dataclass
class BoundaryLayerResult:
    """
    Mock/Stub class representing the outcome of a velocity Boundary Layer solver.
    Since we aren't building the interface right now, this carries essential inputs 
    required for the thermal model.
    """
    arc_length: NDArray          # s coordinates
    U_e: NDArray                 # Edge velocity at boundary
    cf: NDArray                  # Local skin friction coefficient

@dataclass
class ThermalResult:
    """Output of thermal BL computation."""
    arc_length: NDArray             # s coordinates
    nusselt: NDArray                # Local Nusselt number Nu(s)
    heat_transfer_coeff: NDArray    # Local h(s) [W/m^2K]
    wall_heat_flux: NDArray         # Local surface heat flux q(s)
    thermal_bl_thickness: NDArray   # Thermal boundary layer thickness \delta_T(s)
    total_heat_rate: float          # Q = \int q(s) ds [W/m]
    wall_temperature: Optional[NDArray] = None  # Recovered T_w(s) if q_w is given, or matched if T_w is given

@dataclass
class ThermalSolver(ABC):
    """Base thermal BL solver."""
    bl_result: BoundaryLayerResult  # Properties derived from velocity BL
    T_wall: Union[float, NDArray, None] # Wall temperature [K] (uniform or distribution)
    T_inf: float                    # Freestream temperature [K]
    Pr: float                       # Prandtl number
    k: float                        # Thermal conductivity [W/mK]
    q_wall: Union[float, NDArray, None] = None # Alternative Heat flux BC [W/m^2]
    
    def __post_init__(self):
        if self.T_wall is None and self.q_wall is None:
            raise ValueError("Must provide either a wall temperature (T_wall) or a heat flux (q_wall) boundary condition.")
    
    @abstractmethod
    def solve(self) -> ThermalResult:
        ...
