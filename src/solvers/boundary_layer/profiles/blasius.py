"""
Blasius flat-plate velocity profile.

The Blasius similarity solution for a flat plate (zero pressure gradient)
gives constant shape factor and a skin-friction law that depends only on
Re_θ.

    H  = 2.59146   (exact Blasius value)
    cf / 2 = 0.33206 / √Re_θ

This profile ignores the pressure-gradient parameter λ because the
Blasius solution is strictly valid only for dp/dx = 0.  It is useful as
a baseline reference and for validating the BL solver against the
well-known analytical flat-plate solution.

References
----------
* Schlichting, *Boundary-Layer Theory* (8th ed.), §7.3.
* Katz & Plotkin, *Low-Speed Aerodynamics* (2nd ed.), §14.2.
"""

from __future__ import annotations

import math

from .base import VelocityProfile, ProfileClosureData


# Exact Blasius constants
_H_BLASIUS = 2.59146
_CF2_COEFFICIENT = 0.22054  # cf/2 = 0.22054 / Re_theta  (= 0.332 × 0.664)
_DELTA_STAR_RATIO = 0.34361  # δ*/δ (Blasius)
_THETA_RATIO = 0.13268       # θ/δ (Blasius)


class BlasiusProfile(VelocityProfile):
    """
    Blasius (flat-plate) velocity profile closure.

    Provides constant H = 2.591 and cf/2 = 0.2205 / Re_θ regardless of
    the pressure-gradient parameter.  This is the simplest profile and
    serves as a sanity-check baseline.
    """

    @property
    def name(self) -> str:
        return "Blasius"

    def compute_closure(
        self,
        Re_theta: float,
        lambda_param: float = 0.0,
    ) -> ProfileClosureData:
        """
        Evaluate Blasius closure.

        Args:
            Re_theta: Momentum-thickness Reynolds number (must be > 0).
            lambda_param: Ignored (Blasius assumes zero pressure gradient).

        Returns:
            ProfileClosureData with constant H and cf/2 = 0.2205 / Re_θ.
        """
        if Re_theta <= 0:
            raise ValueError(f"Re_theta must be positive, got {Re_theta}")

        cf_2 = _CF2_COEFFICIENT / Re_theta

        return ProfileClosureData(
            H=_H_BLASIUS,
            cf_2=cf_2,
            delta_star_ratio=_DELTA_STAR_RATIO,
            theta_ratio=_THETA_RATIO,
        )

    def initial_theta(self, nu: float, Ue0: float) -> float:
        """
        Blasius starting momentum thickness.

        Uses the Blasius flat-plate formula:
            θ = 0.664 √(ν s₀ / Ue)
        with a small nominal starting distance s₀ = ν / Ue to bootstrap.

        Args:
            nu: Kinematic viscosity [m²/s].
            Ue0: Edge velocity at the first non-singular station [m/s].

        Returns:
            Initial θ₀ [m].
        """
        if Ue0 <= 0:
            raise ValueError(f"Ue0 must be positive, got {Ue0}")

        # Small starting distance to avoid zero θ
        s0 = nu / Ue0
        return 0.664 * math.sqrt(nu * s0 / Ue0)
