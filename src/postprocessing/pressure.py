"""
Pressure field calculation using Bernoulli equation.

Computes:
- Absolute pressure (P)
- Relative/gauge pressure (P - P_ref)
- Pressure coefficient (Cp)
- Total/stagnation pressure (P0)

Supports gravity effects for hydrostatic pressure.
"""

from typing import Set, Optional
import numpy as np
from numpy.typing import NDArray

from .pipeline import PostProcessor
from .fields import FieldData
from .fluid import FluidState, ReferenceType


class PressureProcessor(PostProcessor):
    """
    Compute pressure field from velocity using Bernoulli equation.
    
    Bernoulli (incompressible, steady, inviscid):
        P + 0.5*ρ*V² + ρ*g*y = const
    
    With freestream reference:
        P = P_inf + 0.5*ρ*(V_inf² - V²) - ρ*g*(y - y_ref)
    
    Pressure coefficient:
        Cp = (P - P_inf) / (0.5*ρ*V_inf²) = 1 - (V/V_inf)²
    
    Produces:
        - pressure: Absolute static pressure [Pa]
        - pressure_gauge: Gauge pressure (P - P_ref) [Pa]
        - pressure_coefficient: Cp [-]
        - pressure_total: Stagnation pressure P0 [Pa]
    """
    
    def __init__(self, include_gravity: bool = True):
        """
        Args:
            include_gravity: Include hydrostatic pressure term (ρgh)
        """
        self.include_gravity = include_gravity
    
    @property
    def requires(self) -> Set[str]:
        return {"velocity"}
    
    @property
    def produces(self) -> Set[str]:
        return {"pressure", "pressure_gauge", "pressure_coefficient", "pressure_total"}
    
    def process(self, fields: FieldData, fluid: Optional[FluidState] = None) -> None:
        """Compute pressure fields."""
        if fluid is None:
            raise ValueError("PressureProcessor requires FluidState")
        
        # Get velocity field
        vel = fields.get_vector("velocity")
        if vel is None:
            raise ValueError("Velocity field not found")
        
        V_mag = vel.magnitude  # |V| at each grid point
        
        # Fluid properties
        rho = fluid.density
        p_ref = fluid.reference.pressure
        g = fluid.gravity if self.include_gravity else 0.0
        
        # Reference velocity (freestream)
        v_inf = fluid.reference.velocity
        if v_inf is None:
            # Try to get from metadata
            v_inf = fields.get_metadata("v_inf", 1.0)
        
        # Dynamic pressure
        q_inf = 0.5 * rho * v_inf**2
        
        # Bernoulli: P + 0.5*rho*V² + rho*g*y = P_inf + 0.5*rho*V_inf² + rho*g*y_ref
        # Assuming y_ref = 0 (reference at y=0)
        
        # Static pressure
        P = p_ref + 0.5 * rho * (v_inf**2 - V_mag**2)
        
        # Add gravity term if enabled (y is vertical)
        if g != 0.0 and fields.YY is not None:
            # Hydrostatic: -ρg*y (pressure decreases with height)
            P = P - rho * g * fields.YY
        
        # Gauge pressure (relative to reference)
        P_gauge = P - p_ref
        
        # Pressure coefficient: Cp = (P - P_inf) / q_inf = 1 - (V/V_inf)²
        with np.errstate(divide='ignore', invalid='ignore'):
            Cp = np.where(q_inf > 0, (P - p_ref) / q_inf, 0.0)
            # Equivalent: Cp = 1 - (V_mag / v_inf)**2
        
        # Total/stagnation pressure: P0 = P + 0.5*rho*V²
        P_total = P + 0.5 * rho * V_mag**2
        
        # Add to fields
        fields.add_scalar("pressure", P, units="Pa")
        fields.add_scalar("pressure_gauge", P_gauge, units="Pa")
        fields.add_scalar("pressure_coefficient", Cp, units="")
        fields.add_scalar("pressure_total", P_total, units="Pa")
        
        # Store reference for later
        fields.set_metadata("p_ref", p_ref)
        fields.set_metadata("q_inf", q_inf)


class PressureGradientProcessor(PostProcessor):
    """
    Compute pressure gradient from pressure field.
    
    Produces:
        - pressure_gradient: Vector field (dP/dx, dP/dy)
        - adverse_pressure_gradient: Scalar flag for APG regions
    """
    
    @property
    def requires(self) -> Set[str]:
        return {"pressure", "velocity"}
    
    @property
    def produces(self) -> Set[str]:
        return {"pressure_gradient", "adverse_pressure_gradient"}
    
    def process(self, fields: FieldData, fluid: Optional[FluidState] = None) -> None:
        """Compute pressure gradient."""
        P = fields.get_scalar("pressure")
        vel = fields.get_vector("velocity")
        
        if P is None or vel is None:
            raise ValueError("Required fields not found")
        
        # Grid spacing
        dx = fields.XX[0, 1] - fields.XX[0, 0]
        dy = fields.YY[1, 0] - fields.YY[0, 0]
        
        # Gradient using central differences
        dPdx, dPdy = np.gradient(P.data, dx, dy)
        
        fields.add_vector("pressure_gradient", dPdx, dPdy, units="Pa/m")
        
        # Adverse pressure gradient: dP/ds > 0 in flow direction
        # s = streamwise direction = V/|V|
        V_mag = vel.magnitude
        with np.errstate(divide='ignore', invalid='ignore'):
            sx = np.where(V_mag > 0, vel.u / V_mag, 0)
            sy = np.where(V_mag > 0, vel.v / V_mag, 0)
        
        dPds = dPdx * sx + dPdy * sy  # Streamwise pressure gradient
        
        # Flag APG regions (positive streamwise gradient)
        apg = np.where(dPds > 0, 1.0, 0.0)
        apg[np.isnan(dPds)] = np.nan  # Keep NaN for body interior
        
        fields.add_scalar("adverse_pressure_gradient", apg, units="")
