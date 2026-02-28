"""
Power-law velocity profile (turbulent approximation).

The 1/n power-law profile models turbulent boundary layers:

    u / Ue = (y / δ)^(1/n)

The most common choice is n = 7 (the "1/7th power law"), but n can be
adjusted for different Reynolds-number ranges.

Integral relations for the power law give closed-form H and cf/2 as
functions of n and Re_θ.

    δ*/δ = 1 / (n + 1)
    θ/δ  = n / ((n + 1)(n + 2))
    H    = (n + 2) / n

Skin friction uses the Prandtl-Schlichting empirical correlation:

    cf / 2 = 0.0128 / Re_θ^(1/4)   (n = 7, Blasius friction law)

For general n:

    cf / 2 = a(n) / Re_θ^(1/(n+1))

References
----------
* Schlichting, *Boundary-Layer Theory* (8th ed.), §21.3.
* White, *Viscous Fluid Flow* (3rd ed.), §6-8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import VelocityProfile, ProfileClosureData


@dataclass
class PowerLawProfile(VelocityProfile):
    """
    Power-law (1/n) turbulent velocity profile closure.

    Args:
        n: Power-law exponent (default 7 for the classic 1/7-th law).
    """

    n: int = 7

    @property
    def name(self) -> str:
        return f"Power-law 1/{self.n}"

    def compute_closure(
        self,
        Re_theta: float,
        lambda_param: float = 0.0,
    ) -> ProfileClosureData:
        """
        Evaluate power-law closure.

        The pressure-gradient parameter is ignored because the power-law
        profile does not have a built-in pressure-gradient response.

        Args:
            Re_theta: Momentum-thickness Reynolds number (must be > 0).
            lambda_param: Ignored.

        Returns:
            ProfileClosureData with H and cf/2 for the 1/n law.
        """
        if Re_theta <= 0:
            raise ValueError(f"Re_theta must be positive, got {Re_theta}")

        n = self.n
        delta_star_ratio = 1.0 / (n + 1)
        theta_ratio = n / ((n + 1) * (n + 2))
        H = (n + 2.0) / n  # = delta_star_ratio / theta_ratio

        # Empirical skin-friction law: cf/2 = a / Re_θ^(1/(n+1))
        # For n = 7: a ≈ 0.0128  (Blasius friction law)
        # General: a ≈ 0.0225 · (2 / (n + 1))^(2/(n+1))
        exponent = 1.0 / (n + 1)
        if n == 7:
            a_coeff = 0.0128
        else:
            a_coeff = 0.0225 * (2.0 / (n + 1)) ** (2.0 * exponent)

        cf_2 = a_coeff / (Re_theta ** exponent)

        return ProfileClosureData(
            H=H,
            cf_2=cf_2,
            delta_star_ratio=delta_star_ratio,
            theta_ratio=theta_ratio,
        )

    def initial_theta(self, nu: float, Ue0: float) -> float:
        """
        Starting θ using a turbulent flat-plate correlation.

        θ/x = 0.036 / Rex^(1/5)  (Prandtl 1/7-th law).
        We use a small starter x₀ = 100 ν / Ue₀ to get a finite θ₀.

        Args:
            nu: Kinematic viscosity [m²/s].
            Ue0: Edge velocity at starting station [m/s].
        """
        if Ue0 <= 0:
            raise ValueError(f"Ue0 must be positive, got {Ue0}")
        x0 = 100.0 * nu / Ue0
        Rex0 = Ue0 * x0 / nu
        return 0.036 * x0 / (Rex0 ** 0.2)
