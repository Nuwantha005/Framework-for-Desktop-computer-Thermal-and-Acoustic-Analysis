"""
Fluid state and reference conditions.

Provides a clean way to specify fluid properties without polluting
the case file structure. Properties can be optional - only what's
needed for the current solver is required.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple
from enum import Enum
import numpy as np


class ReferenceType(Enum):
    """Type of reference condition for pressure calculation."""
    FREESTREAM = "freestream"   # P_ref at freestream (external flow)
    OUTLET = "outlet"          # P_ref at outlet (internal flow)
    ABSOLUTE = "absolute"      # Use absolute values


@dataclass
class ReferenceCondition:
    """
    Reference conditions for pressure calculation.
    
    For external flows (airfoils, bluff bodies):
        - Use FREESTREAM: P_ref is the static pressure at infinity
        - Bernoulli gives: P + 0.5*rho*V² = P_inf + 0.5*rho*V_inf²
    
    For internal flows (ducts, diffusers):
        - Use OUTLET: P_ref is specified at outlet
        - Useful when outlet pressure is known (atmospheric)
    """
    type: ReferenceType = ReferenceType.FREESTREAM
    pressure: float = 101325.0  # Pa (1 atm default)
    velocity: Optional[float] = None  # m/s (if type=FREESTREAM, uses flow v_inf)
    location: Optional[Tuple[float, float]] = None  # For outlet type
    
    @classmethod
    def freestream(cls, p_inf: float = 101325.0) -> ReferenceCondition:
        """Create freestream reference (external flow)."""
        return cls(type=ReferenceType.FREESTREAM, pressure=p_inf)
    
    @classmethod
    def outlet(cls, p_outlet: float = 101325.0) -> ReferenceCondition:
        """Create outlet reference (internal flow)."""
        return cls(type=ReferenceType.OUTLET, pressure=p_outlet)


@dataclass
class FluidState:
    """
    Fluid properties for post-processing calculations.
    
    Only density is required for basic pressure calculation.
    Other properties are optional and used by specific processors:
    - viscosity: for boundary layer calculations
    - thermal_conductivity, specific_heat: for thermal analysis
    - gravity: for hydrostatic pressure effects
    
    Usage:
        # Minimal for pressure calculation
        fluid = FluidState(density=1.225)
        
        # With gravity effects
        fluid = FluidState(density=1.225, gravity=-9.81)
        
        # Full properties for thermal BL
        fluid = FluidState.air_standard()
        
        # From case file
        fluid = FluidState.from_dict(case_config['fluid'])
    """
    
    # Required for pressure
    density: float  # kg/m³
    
    # Reference condition
    reference: ReferenceCondition = field(default_factory=ReferenceCondition.freestream)
    
    # Optional: gravity (for hydrostatic effects)
    gravity: float = 0.0  # m/s² (negative = downward in -y direction)
    
    # Optional: for viscous calculations
    viscosity: Optional[float] = None  # Pa·s (dynamic viscosity)
    
    # Optional: for thermal calculations  
    thermal_conductivity: Optional[float] = None  # W/(m·K)
    specific_heat_cp: Optional[float] = None  # J/(kg·K)
    specific_heat_cv: Optional[float] = None  # J/(kg·K)
    
    # Optional: for compressible flows
    speed_of_sound: Optional[float] = None  # m/s
    
    @property
    def kinematic_viscosity(self) -> Optional[float]:
        """ν = μ/ρ (m²/s)"""
        if self.viscosity is None:
            return None
        return self.viscosity / self.density
    
    @property
    def prandtl_number(self) -> Optional[float]:
        """Pr = μ·Cp/k"""
        if None in (self.viscosity, self.specific_heat_cp, self.thermal_conductivity):
            return None
        return self.viscosity * self.specific_heat_cp / self.thermal_conductivity
    
    @property
    def gamma(self) -> Optional[float]:
        """γ = Cp/Cv"""
        if None in (self.specific_heat_cp, self.specific_heat_cv):
            return None
        return self.specific_heat_cp / self.specific_heat_cv
    
    # -------------------------------------------------------------------------
    # Standard fluids
    # -------------------------------------------------------------------------
    
    @classmethod
    def air_standard(cls, p_ref: float = 101325.0) -> FluidState:
        """Air at standard conditions (15°C, 1 atm)."""
        return cls(
            density=1.225,
            reference=ReferenceCondition.freestream(p_ref),
            viscosity=1.789e-5,
            thermal_conductivity=0.0253,
            specific_heat_cp=1005.0,
            specific_heat_cv=718.0,
            speed_of_sound=340.3
        )
    
    @classmethod
    def water_standard(cls, p_ref: float = 101325.0) -> FluidState:
        """Water at standard conditions (20°C, 1 atm)."""
        return cls(
            density=998.2,
            reference=ReferenceCondition.freestream(p_ref),
            viscosity=1.002e-3,
            thermal_conductivity=0.598,
            specific_heat_cp=4182.0,
            specific_heat_cv=4182.0,  # Incompressible
            speed_of_sound=1481.0
        )
    
    @classmethod
    def incompressible(cls, density: float, p_ref: float = 101325.0) -> FluidState:
        """Simple incompressible fluid with just density."""
        return cls(
            density=density,
            reference=ReferenceCondition.freestream(p_ref)
        )
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> dict:
        """Convert to dictionary for case file export."""
        d = {
            "density": self.density,
            "reference": {
                "type": self.reference.type.value,
                "pressure": self.reference.pressure
            }
        }
        
        if self.gravity != 0.0:
            d["gravity"] = self.gravity
        if self.viscosity is not None:
            d["viscosity"] = self.viscosity
        if self.thermal_conductivity is not None:
            d["thermal_conductivity"] = self.thermal_conductivity
        if self.specific_heat_cp is not None:
            d["specific_heat_cp"] = self.specific_heat_cp
        if self.specific_heat_cv is not None:
            d["specific_heat_cv"] = self.specific_heat_cv
        if self.speed_of_sound is not None:
            d["speed_of_sound"] = self.speed_of_sound
            
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> FluidState:
        """Create from dictionary (case file)."""
        ref_data = data.get("reference", {})
        ref_type = ReferenceType(ref_data.get("type", "freestream"))
        ref = ReferenceCondition(
            type=ref_type,
            pressure=ref_data.get("pressure", 101325.0)
        )
        
        return cls(
            density=data["density"],
            reference=ref,
            gravity=data.get("gravity", 0.0),
            viscosity=data.get("viscosity"),
            thermal_conductivity=data.get("thermal_conductivity"),
            specific_heat_cp=data.get("specific_heat_cp"),
            specific_heat_cv=data.get("specific_heat_cv"),
            speed_of_sound=data.get("speed_of_sound")
        )
    
    def __repr__(self) -> str:
        parts = [f"ρ={self.density} kg/m³"]
        if self.viscosity:
            parts.append(f"μ={self.viscosity:.2e} Pa·s")
        if self.gravity:
            parts.append(f"g={self.gravity} m/s²")
        return f"FluidState({', '.join(parts)})"
