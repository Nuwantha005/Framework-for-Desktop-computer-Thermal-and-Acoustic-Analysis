"""
Reynolds Analogy thermal boundary layer solver.

Uses the Chilton-Colburn analogy to derive heat transfer coefficients
directly from skin friction. Fast baseline method suitable for attached
boundary layers with moderate pressure gradients.

Theory
------
The Chilton-Colburn analogy relates momentum and heat transfer:

    St = (cf / 2) * Pr^(-2/3)

where St is the Stanton number, cf the skin friction coefficient, and
Pr the Prandtl number. The local heat transfer coefficient is:

    h = St * ρ * cp * Ue

and the Nusselt number:

    Nu = h * L / k

References
----------
* Kays & Crawford, "Convective Heat and Mass Transfer", 4th ed.
* Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer", Ch. 7.
"""

import numpy as np
from numpy.typing import NDArray

from .base import (
    ThermalSolver,
    ThermalBLInput,
    ThermalSolverConfig,
    ThermalResult,
)
from .utils import compute_total_heat_rate


class ReynoldsAnalogyThermal(ThermalSolver):
    """
    Thermal boundary layer solver using Chilton-Colburn Reynolds Analogy.
    
    Fast baseline method that derives Nu directly from cf. Suitable for:
    - Attached boundary layers
    - Moderate pressure gradients
    - Quick estimates before running BDIM
    
    Example::
    
        from solvers.thermal import ReynoldsAnalogyThermal
        from solvers.thermal.base import ThermalBLInput, ThermalSolverConfig
        
        # Input from viscous BL
        bl_input = ThermalBLInput(
            side="upper",
            arc_length=s,
            x=x, y=y,
            Ue=Ue,
            cf=cf,
            delta=delta,
        )
        
        # Config with heat flux BC
        config = ThermalSolverConfig(
            T_inf=300.0,
            q_wall=1000.0,  # W/m²
            Pr=0.71,
            k=0.026,
        )
        
        solver = ReynoldsAnalogyThermal(bl_input, config)
        result = solver.solve()
    """
    
    @property
    def name(self) -> str:
        return "reynolds_analogy"
    
    def solve(self) -> ThermalResult:
        """
        Compute thermal BL solution using Chilton-Colburn analogy.
        
        Returns:
            ThermalResult with T_w, h, Nu, q_w, δ_T, and total Q.
        """
        s = self.bl_input.arc_length
        Ue = self.bl_input.Ue
        cf = self.bl_input.cf
        n = len(s)
        
        # Chilton-Colburn analogy: St = (cf / 2) * Pr^(-2/3)
        stanton = (cf / 2.0) * (self.config.Pr ** (-2.0 / 3.0))
        
        # Heat transfer coefficient: h = St * ρ * cp * |Ue|
        h = stanton * self.config.rho * self.config.cp * np.abs(Ue)
        
        # Handle boundary conditions
        if self.config.q_wall is not None:
            # Neumann BC: heat flux given, solve for wall temperature
            q_w = self._expand_bc(self.config.q_wall, n)
            # T_w = T_inf + q_w / h (safe divide for low h)
            T_w = self.config.T_inf + np.divide(
                q_w, h,
                out=np.full_like(h, self.config.T_inf),
                where=(h > 1e-10)
            )
        else:
            # Dirichlet BC: wall temperature given, compute heat flux
            T_w = self._expand_bc(self.config.T_wall, n)
            q_w = h * (T_w - self.config.T_inf)
        
        # Nusselt number: Nu = h * L_char / k
        L_char = float(np.max(s)) if len(s) > 0 else 1.0
        nusselt = (h * L_char) / self.config.k
        
        # Thermal BL thickness: δ_T ≈ δ / Pr^(1/3)
        if self.bl_input.has_delta:
            delta_T = self.bl_input.delta / (self.config.Pr ** (1.0 / 3.0))
        else:
            # Fallback: estimate from theta and H if available
            delta_T = np.zeros_like(s)
            if self.bl_input.theta is not None and self.bl_input.H is not None:
                # δ* = θ * H, δ ≈ δ* / 0.35, δ_T ≈ δ / Pr^(1/3)
                delta_star = self.bl_input.theta * self.bl_input.H
                delta_est = delta_star / 0.35
                delta_T = delta_est / (self.config.Pr ** (1.0 / 3.0))
        
        # Total heat rate: Q = ∫ q_w ds [W/m per unit span]
        total_Q = compute_total_heat_rate(q_w, s)
        
        return ThermalResult(
            side=self.bl_input.side,
            arc_length=s,
            x=self.bl_input.x,
            y=self.bl_input.y,
            wall_temperature=T_w,
            heat_transfer_coeff=h,
            nusselt=nusselt,
            wall_heat_flux=q_w,
            thermal_bl_thickness=delta_T,
            total_heat_rate=total_Q,
            solver_type=self.name,
        )
    
    def _expand_bc(
        self,
        bc_value: float | NDArray,
        n: int,
    ) -> NDArray[np.float64]:
        """Expand scalar BC to array if needed."""
        if isinstance(bc_value, (int, float)):
            return np.full(n, bc_value, dtype=np.float64)
        else:
            arr = np.asarray(bc_value, dtype=np.float64)
            if len(arr) != n:
                raise ValueError(
                    f"BC array length {len(arr)} doesn't match stations {n}"
                )
            return arr
