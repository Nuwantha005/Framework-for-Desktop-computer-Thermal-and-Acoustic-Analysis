"""
Thermal boundary layer solver base classes and data structures.

Provides the common interface for thermal solvers (Reynolds Analogy, BDIM)
and defines the input/output data structures for coupling with the viscous
boundary layer solver.

The key data flow is:
    Viscous BL (BoundaryLayerResult) → ThermalBLInput → ThermalSolver → ThermalResult

ThermalBLInput is the common interface that all thermal solvers accept,
extracted from the viscous BL result. This allows different thermal solvers
to be swapped without changing the upstream BL computation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Input interface: viscous BL → thermal solver
# ---------------------------------------------------------------------------

@dataclass
class ThermalBLInput:
    """
    Input data for thermal solver, extracted from viscous boundary layer.
    
    This is the common interface between viscous BL and thermal solvers.
    Contains only the data needed for thermal calculations, with NaN values
    filtered out (only valid region before separation).
    
    Attributes:
        side: Surface side identifier ("upper" or "lower")
        arc_length: Arc-length coordinates [m], shape (K,). Monotonic, starting
            from near stagnation. Only valid (non-separated) stations.
        x: Surface x-coordinates [m], shape (K,).
        y: Surface y-coordinates [m], shape (K,).
        Ue: Edge velocity [m/s], shape (K,). Always positive.
        cf: Skin friction coefficient [-], shape (K,). Freestream-normalized.
        delta: Boundary layer thickness [m], shape (K,). Optional, for δ_T estimation.
        theta: Momentum thickness [m], shape (K,). Optional.
        H: Shape factor [-], shape (K,). Optional.
        nu: Kinematic viscosity [m²/s] used in the BL computation.
    """
    side: str
    arc_length: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    Ue: NDArray[np.float64]
    cf: NDArray[np.float64]
    delta: Optional[NDArray[np.float64]] = None
    theta: Optional[NDArray[np.float64]] = None
    H: Optional[NDArray[np.float64]] = None
    nu: float = 1.5e-5
    
    def __post_init__(self):
        """Validate input arrays."""
        n = len(self.arc_length)
        if len(self.Ue) != n or len(self.cf) != n:
            raise ValueError(
                f"Array length mismatch: arc_length={n}, Ue={len(self.Ue)}, cf={len(self.cf)}"
            )
        if len(self.x) != n or len(self.y) != n:
            raise ValueError(
                f"Coordinate length mismatch: arc_length={n}, x={len(self.x)}, y={len(self.y)}"
            )
    
    @property
    def num_stations(self) -> int:
        """Number of valid arc-length stations."""
        return len(self.arc_length)
    
    @property
    def has_delta(self) -> bool:
        """Whether BL thickness data is available."""
        return self.delta is not None and len(self.delta) > 0


def extract_thermal_input(
    bl_path_result,  # BoundaryLayerPathResult from runner
    profile_name: str,
) -> ThermalBLInput:
    """
    Extract thermal solver input from a viscous BL path result.
    
    Filters out NaN values (stagnation skip and separation regions) to provide
    only the valid region where thermal calculations make sense.
    
    Args:
        bl_path_result: BoundaryLayerPathResult from BoundaryLayerRunner
        profile_name: Which velocity profile result to use
    
    Returns:
        ThermalBLInput with valid (non-NaN) stations only
    
    Raises:
        KeyError: If profile_name not found in results
        ValueError: If no valid stations remain after filtering
    """
    if profile_name not in bl_path_result.results:
        raise KeyError(
            f"Profile '{profile_name}' not found. "
            f"Available: {list(bl_path_result.results.keys())}"
        )
    
    bl_result = bl_path_result.results[profile_name]
    
    # Find valid (non-NaN) stations based on theta (momentum thickness)
    valid_mask = ~np.isnan(bl_result.theta)
    
    if not np.any(valid_mask):
        raise ValueError(
            f"No valid stations for profile '{profile_name}' on {bl_path_result.side} side. "
            "BL may have separated immediately."
        )
    
    # Extract valid stations
    s_valid = bl_result.s[valid_mask]
    Ue_valid = bl_result.Ue[valid_mask]
    cf_valid = bl_result.cf[valid_mask]
    theta_valid = bl_result.theta[valid_mask]
    H_valid = bl_result.H[valid_mask]
    
    # Get coordinates from path result (same indices)
    x_valid = bl_path_result.x[valid_mask]
    y_valid = bl_path_result.y[valid_mask]
    
    # Compute delta if field data is available
    delta_valid = None
    if profile_name in bl_path_result.fields:
        field = bl_path_result.fields[profile_name]
        # field.delta is already at valid stations only
        # Need to match indices - field may have different masking
        if len(field.delta) == len(s_valid):
            delta_valid = field.delta
    
    # If no field data, estimate delta from theta and H
    # Using delta* = theta * H, and delta ~ delta* / (ratio depends on profile)
    # For Blasius: delta*/delta ~ 0.344, so delta ~ delta* / 0.344 ~ 2.9 * delta*
    # This is approximate but sufficient for thermal BL thickness estimation
    if delta_valid is None and theta_valid is not None and H_valid is not None:
        delta_star = theta_valid * H_valid
        delta_valid = delta_star / 0.35  # Approximate for laminar profiles
    
    return ThermalBLInput(
        side=bl_path_result.side,
        arc_length=s_valid,
        x=x_valid,
        y=y_valid,
        Ue=Ue_valid,
        cf=cf_valid,
        delta=delta_valid,
        theta=theta_valid,
        H=H_valid,
        nu=0.0,  # Will be set by caller if needed
    )


# ---------------------------------------------------------------------------
# Output: thermal solver result
# ---------------------------------------------------------------------------

@dataclass
class ThermalFieldData:
    """
    Domain temperature field data from BDIM thermal solver.
    
    Contains the temperature field in both physical (x, y) and
    boundary layer (s, y_normal) coordinates for visualization.
    
    Attributes:
        s: Arc-length stations [m], shape (M,)
        y_normal: Wall-normal distances [m], shape (M, Ny)
        x: Physical x-coordinates [m], shape (M, Ny)
        y: Physical y-coordinates [m], shape (M, Ny)
        T: Temperature field [K], shape (M, Ny)
        T_inf: Freestream temperature [K] for normalization
        side: Surface side identifier
    """
    s: NDArray[np.float64]
    y_normal: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    T: NDArray[np.float64]
    T_inf: float = 300.0
    side: str = ""
    
    @property
    def num_stations(self) -> int:
        """Number of arc-length stations."""
        return len(self.s)
    
    @property
    def num_y_points(self) -> int:
        """Number of wall-normal points per station."""
        return self.y_normal.shape[1] if self.y_normal.ndim == 2 else 0
    
    @property
    def T_normalized(self) -> NDArray[np.float64]:
        """Normalized temperature (T - T_inf) / (T_wall - T_inf)."""
        T_wall = self.T[:, 0]  # Wall temperature is at y=0
        denom = T_wall[:, np.newaxis] - self.T_inf
        denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)
        return (self.T - self.T_inf) / denom


@dataclass
class ThermalResult:
    """
    Output of thermal boundary layer computation for one surface side.
    
    All arrays share length K (number of valid arc-length stations).
    
    Attributes:
        side: Surface side identifier ("upper" or "lower")
        arc_length: Arc-length coordinates [m], shape (K,).
        x: Surface x-coordinates [m], shape (K,).
        y: Surface y-coordinates [m], shape (K,).
        wall_temperature: Wall temperature T_w(s) [K], shape (K,).
            Computed if heat flux BC given, or input if temperature BC given.
        heat_transfer_coeff: Local heat transfer coefficient h(s) [W/m²K], shape (K,).
        nusselt: Local Nusselt number Nu(s) [-], shape (K,).
        wall_heat_flux: Surface heat flux q_w(s) [W/m²], shape (K,).
            Input if Neumann BC, computed if Dirichlet BC.
        thermal_bl_thickness: Thermal BL thickness δ_T(s) [m], shape (K,).
        total_heat_rate: Integrated heat transfer Q = ∫q_w ds [W/m per unit span].
        solver_type: Name of the thermal solver used.
        field: Optional domain temperature field (only from BDIM solver).
    """
    side: str
    arc_length: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    wall_temperature: NDArray[np.float64]
    heat_transfer_coeff: NDArray[np.float64]
    nusselt: NDArray[np.float64]
    wall_heat_flux: NDArray[np.float64]
    thermal_bl_thickness: NDArray[np.float64]
    total_heat_rate: float
    solver_type: str = ""
    field: Optional[ThermalFieldData] = None
    
    @property
    def num_stations(self) -> int:
        """Number of arc-length stations."""
        return len(self.arc_length)
    
    @property
    def has_field(self) -> bool:
        """Whether domain field data is available."""
        return self.field is not None


# ---------------------------------------------------------------------------
# Thermal solver base class
# ---------------------------------------------------------------------------

@dataclass
class ThermalSolverConfig:
    """
    Configuration for thermal solver.
    
    Attributes:
        T_inf: Freestream temperature [K].
        Pr: Prandtl number [-]. Default ~0.71 for air.
        k: Thermal conductivity [W/mK]. Default ~0.026 for air at STP.
        rho: Fluid density [kg/m³]. Default 1.225 for air at STP.
        cp: Specific heat at constant pressure [J/kgK]. Default 1005 for air.
        q_wall: Heat flux BC [W/m²]. If provided, solve for T_wall.
        T_wall: Wall temperature BC [K]. If provided, solve for q_wall.
    """
    T_inf: float = 300.0
    Pr: float = 0.71
    k: float = 0.026
    rho: float = 1.225
    cp: float = 1005.0
    q_wall: Optional[Union[float, NDArray[np.float64]]] = None
    T_wall: Optional[Union[float, NDArray[np.float64]]] = None
    
    def __post_init__(self):
        if self.T_inf <= 0:
            raise ValueError(f"Freestream temperature must be positive, got {self.T_inf}")
        if self.q_wall is None and self.T_wall is None:
            raise ValueError(
                "Must provide either q_wall (heat flux) or T_wall (temperature) BC"
            )


class ThermalSolver(ABC):
    """
    Abstract base class for thermal boundary layer solvers.
    
    All thermal solvers accept ThermalBLInput and ThermalSolverConfig,
    and return ThermalResult. This allows different solvers (Reynolds
    Analogy, BDIM, etc.) to be swapped easily.
    
    Example usage::
    
        from solvers.thermal import ReynoldsAnalogyThermal, ThermalSolverConfig
        from solvers.thermal.base import extract_thermal_input
        
        # Get thermal input from viscous BL result
        thermal_input = extract_thermal_input(bl_path_result, "thwaites")
        
        # Configure and run thermal solver
        config = ThermalSolverConfig(T_inf=300.0, q_wall=1000.0, Pr=0.71, k=0.026)
        solver = ReynoldsAnalogyThermal(thermal_input, config)
        result = solver.solve()
        
        print(f"Total heat rate: {result.total_heat_rate:.2f} W/m")
    """
    
    def __init__(self, bl_input: ThermalBLInput, config: ThermalSolverConfig):
        """
        Initialize thermal solver.
        
        Args:
            bl_input: Input from viscous BL solver (ThermalBLInput)
            config: Thermal solver configuration
        """
        self.bl_input = bl_input
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Solver name identifier."""
        ...
    
    @abstractmethod
    def solve(self) -> ThermalResult:
        """
        Compute thermal boundary layer solution.
        
        Returns:
            ThermalResult with temperature, heat transfer coefficients, etc.
        """
        ...
