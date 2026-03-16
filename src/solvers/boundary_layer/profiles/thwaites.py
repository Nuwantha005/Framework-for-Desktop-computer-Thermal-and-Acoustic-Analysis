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
from typing import Literal

import numpy as np
from numpy.typing import NDArray

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

    Since Thwaites provides no explicit velocity profile shape, velocity
    reconstruction is delegated to a paired profile family.

    Args:
        reconstruction: Which profile family to use for velocity field
            reconstruction.  ``"falkner_skan"`` (default) inverts H → β
            and uses the F-S similarity profile.  ``"pohlhausen"`` inverts
            H → Λ and uses the 4th-order polynomial.
    """

    def __init__(
        self,
        reconstruction: Literal["falkner_skan", "pohlhausen"] = "falkner_skan",
    ) -> None:
        self._reconstruction = reconstruction

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

    def stagnation_theta(self, nu: float, K: float) -> float:
        """
        Thwaites stagnation patching: θ_stag = √(0.075 · ν / K).

        Thwaites' quadrature θ² = 0.45ν/Ue⁶ ∫ Ue⁵ ds with Ue = K·s
        evaluates analytically at the stagnation point to give
        C = 0.45/6 = 0.075.

        Args:
            nu: Kinematic viscosity [m²/s].
            K: Velocity gradient dUe/ds at stagnation [1/s].

        Returns:
            Stagnation momentum thickness θ_stag [m].
        """
        self._validate_stagnation_args(nu, K)
        return math.sqrt(0.075 * nu / K)

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

    # ------------------------------------------------------------------
    # Post-processing: velocity field reconstruction (via pairing)
    # ------------------------------------------------------------------

    @property
    def reconstruction(self) -> Literal["falkner_skan", "pohlhausen"]:
        """Active reconstruction pairing."""
        return self._reconstruction  # type: ignore[return-value]

    def _paired_profile(self) -> VelocityProfile:
        """Return the paired profile instance used for reconstruction."""
        if self._reconstruction == "pohlhausen":
            from .pohlhausen import PohlhausenProfile
            return PohlhausenProfile()
        else:
            from .falkner_skan import FalknerSkanProfile
            return FalknerSkanProfile()

    def compute_delta(self, theta: float, H: float) -> float:
        """
        Compute δ by delegating to the paired reconstruction profile.

        Args:
            theta: Momentum thickness θ [m].
            H: Shape factor δ*/θ.

        Returns:
            Boundary layer thickness [m].
        """
        return self._paired_profile().compute_delta(theta, H)

    def reconstruct_velocity(
        self,
        y: NDArray[np.float64],
        theta: float,
        H: float,
        Ue: float,
    ) -> NDArray[np.float64]:
        """
        Reconstruct u(y) by delegating to the paired profile.

        Args:
            y: Wall-normal coordinates [m], shape (Ny,).
            theta: Momentum thickness θ [m].
            H: Shape factor δ*/θ.
            Ue: Edge velocity [m/s].

        Returns:
            Velocity u(y) [m/s], shape (Ny,).
        """
        return self._paired_profile().reconstruct_velocity(y, theta, H, Ue)
