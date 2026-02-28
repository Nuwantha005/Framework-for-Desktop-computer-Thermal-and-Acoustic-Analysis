"""
Thwaites' one-parameter integral method.

Thwaites' method is a simple, well-tested correlation for laminar boundary
layers that avoids choosing an explicit velocity profile.  It uses the
single parameter:

    λ = (θ² / ν) dUe/ds

and tabulated/correlated functions S(λ) and H(λ) derived from exact
Falkner-Skan solutions:

    cf/2 = S(λ) · ν / (Ue θ)  =  S(λ) / Re_θ
    H    = H(λ)

Additionally, Thwaites gives a closed-form solution for θ²(s):

    θ²(s) = (0.45 ν / Ue(s)⁶) ∫₀ˢ Ue(s')⁵ ds'

which can be evaluated by quadrature, bypassing ODE integration entirely.
The ``BoundaryLayerSolver`` still integrates the ODE (for generality), but
the ``initial_theta`` from Thwaites' quadrature provides an excellent
starting value.

Separation is predicted at λ ≈ −0.09.

References
----------
* Thwaites, B., "Approximate Calculation of the Laminar Boundary Layer",
  *Aero. Quarterly* 1, 245–280, 1949.
* Katz & Plotkin, *Low-Speed Aerodynamics* (2nd ed.), §14.3.
* White, *Viscous Fluid Flow* (3rd ed.), §4-6.2.
"""

from __future__ import annotations

import math

import numpy as np

from .base import VelocityProfile, ProfileClosureData


# Thwaites' correlation tables (λ → H, S)
# Fitted from Falkner-Skan family data (White Table 4-4)
_TH_LAMBDA = np.array([
    -0.09, -0.08, -0.06, -0.04, -0.02,
     0.0,   0.02,  0.04,  0.06,  0.08,
     0.10,  0.15,  0.20,  0.25,
])
_TH_H = np.array([
    3.55,  3.49,  3.22,  2.99,  2.81,
    2.61,  2.47,  2.34,  2.23,  2.15,
    2.08,  1.95,  1.85,  1.78,
])
_TH_S = np.array([
    0.0,   0.015, 0.072, 0.132, 0.190,
    0.220, 0.268, 0.306, 0.338, 0.366,
    0.390, 0.440, 0.478, 0.508,
])


class ThwaitesProfile(VelocityProfile):
    """
    Thwaites' one-parameter laminar BL correlation.

    Uses tabulated S(λ) and H(λ) derived from Falkner-Skan solutions.
    Separation is flagged when λ < −0.09.
    """

    @property
    def name(self) -> str:
        return "Thwaites"

    def compute_closure(
        self,
        Re_theta: float,
        lambda_param: float = 0.0,
    ) -> ProfileClosureData:
        """
        Evaluate Thwaites closure at a single station.

        Args:
            Re_theta: Momentum-thickness Reynolds number.
            lambda_param: Thwaites parameter λ = θ²/ν · dUe/ds.

        Returns:
            ProfileClosureData with H(λ) and cf/2 = S(λ)/Re_θ.
        """
        if Re_theta <= 0:
            raise ValueError(f"Re_theta must be positive, got {Re_theta}")

        lam = max(_TH_LAMBDA[0], min(_TH_LAMBDA[-1], lambda_param))

        H = float(np.interp(lam, _TH_LAMBDA, _TH_H))
        S = float(np.interp(lam, _TH_LAMBDA, _TH_S))

        cf_2 = S / Re_theta if Re_theta > 0 else 0.0

        return ProfileClosureData(H=H, cf_2=cf_2)

    def initial_theta(self, nu: float, Ue0: float) -> float:
        """
        Thwaites' quadrature starting value.

        Uses the closed-form expression for the first station:
            θ₀² = 0.45 ν / Ue₀⁶ · (Ue₀⁵ · s₀)  →  θ₀ = √(0.45 ν s₀ / Ue₀)

        with s₀ = ν / Ue₀ as a small bootstrap distance.

        Args:
            nu: Kinematic viscosity [m²/s].
            Ue0: Edge velocity at starting station [m/s].
        """
        if Ue0 <= 0:
            raise ValueError(f"Ue0 must be positive, got {Ue0}")
        s0 = nu / Ue0
        return math.sqrt(0.45 * nu * s0 / Ue0)

    # ------------------------------------------------------------------
    # Utility: direct Thwaites quadrature (can be used externally)
    # ------------------------------------------------------------------

    @staticmethod
    def quadrature_theta(
        s: np.ndarray,
        Ue: np.ndarray,
        nu: float,
    ) -> np.ndarray:
        """
        Compute θ(s) directly via Thwaites' quadrature formula.

        θ²(s) = (0.45 ν / Ue(s)⁶) ∫₀ˢ Ue(s')⁵ ds'

        This bypasses the ODE integrator and gives the exact Thwaites
        solution.  Useful for comparison / validation.

        Args:
            s: Arc-length stations, shape (M,).
            Ue: Edge velocity at each station [m/s], shape (M,).
            nu: Kinematic viscosity [m²/s].

        Returns:
            Momentum thickness θ at each station, shape (M,).
        """
        integral = np.zeros_like(s)
        Ue5 = Ue ** 5
        for i in range(1, len(s)):
            ds = s[i] - s[i - 1]
            integral[i] = integral[i - 1] + 0.5 * (Ue5[i] + Ue5[i - 1]) * ds

        Ue6 = Ue ** 6
        # Avoid division by zero at stagnation
        safe_Ue6 = np.where(np.abs(Ue6) > 1e-30, Ue6, 1e-30)
        theta_sq = 0.45 * nu / safe_Ue6 * integral
        return np.sqrt(np.maximum(theta_sq, 0.0))
