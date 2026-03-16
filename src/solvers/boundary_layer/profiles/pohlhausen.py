"""
Pohlhausen 4th-order polynomial velocity profile.

The Kármán–Pohlhausen method approximates the laminar boundary-layer
velocity profile with a 4th-degree polynomial in η = y/δ:

    u/Ue = 2η − 2η³ + η⁴ + (Λ/6)(η − 3η² + 3η³ − η⁴)

where Λ = (δ²/ν)(dUe/ds) is the pressure-gradient parameter.  The
profile satisfies the wall and edge boundary conditions exactly and
incorporates the effect of a streamwise pressure gradient through Λ.

From the polynomial, closed-form expressions for H and cf/2 as
functions of Λ are obtained by integrating across the layer.

Valid range: −12 ≤ Λ ≤ 12  (separation at Λ = −12, where cf = 0).

References
----------
* Schlichting, *Boundary-Layer Theory* (8th ed.), §8.2.
* Katz & Plotkin, *Low-Speed Aerodynamics* (2nd ed.), §14.3.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .base import VelocityProfile, ProfileClosureData


class PohlhausenProfile(VelocityProfile):
    """
    Pohlhausen 4th-order polynomial velocity profile closure.

    Closure quantities are polynomial functions of the Pohlhausen
    pressure-gradient parameter Λ = (δ²/ν)(dUe/ds).

    Note: The BL solver passes λ_Thwaites = θ²/ν · dUe/ds.  The
    Pohlhausen Λ relates to it via the momentum-thickness ratio, so an
    internal conversion is performed.
    """

    @property
    def name(self) -> str:
        return "Pohlhausen"

    def compute_closure(
        self,
        Re_theta: float,
        lambda_param: float = 0.0,
    ) -> ProfileClosureData:
        """
        Evaluate Pohlhausen closure given Re_θ and λ.

        The solver supplies λ = θ²/ν · dUe/ds (Thwaites convention).
        We convert internally to the Pohlhausen Λ ≈ λ / (θ/δ)² using
        the profile's own θ/δ relation, iterated once.

        Args:
            Re_theta: Momentum-thickness Reynolds number.
            lambda_param: Thwaites-style parameter θ²/ν · dUe/ds.

        Returns:
            ProfileClosureData with H and cf/2 from the polynomial profile.
        """
        if Re_theta <= 0:
            raise ValueError(f"Re_theta must be positive, got {Re_theta}")

        # First estimate: use zero-pressure-gradient ratios to bootstrap Λ
        # θ/δ for Λ=0 is 37/315 ≈ 0.11746
        theta_delta_0 = 37.0 / 315.0
        Lambda = lambda_param / (theta_delta_0 ** 2) if abs(theta_delta_0) > 1e-15 else 0.0

        # Clamp to valid range
        Lambda = max(-12.0, min(12.0, Lambda))

        # Refine once: get θ/δ at this Λ, re-estimate
        theta_delta = self._theta_delta(Lambda)
        if abs(theta_delta) > 1e-15:
            Lambda = lambda_param / (theta_delta ** 2)
            Lambda = max(-12.0, min(12.0, Lambda))
            theta_delta = self._theta_delta(Lambda)

        delta_star_delta = self._delta_star_delta(Lambda)
        H = delta_star_delta / theta_delta if abs(theta_delta) > 1e-15 else 2.59

        # Wall shear: cf/2 = ν / (Ue δ) · (du/dy)|_{y=0} / Ue
        # From the polynomial: (du/dy)|_{y=0} / (Ue/δ) = 2 + Λ/6
        f_wall = 2.0 + Lambda / 6.0
        # cf/2 = f_wall · θ / (θ Re_θ) using δ = θ / theta_delta
        if abs(theta_delta) > 1e-15 and Re_theta > 0:
            cf_2 = f_wall * theta_delta / Re_theta
        else:
            cf_2 = 0.0

        return ProfileClosureData(
            H=H,
            cf_2=cf_2,
            delta_star_ratio=delta_star_delta,
            theta_ratio=theta_delta,
        )

    def initial_theta(self, nu: float, Ue0: float) -> float:
        """
        Starting θ using the zero-pressure-gradient Pohlhausen result.

        For Λ = 0 the Pohlhausen profile coincides with the standard
        4th-order flat-plate approximation giving θ/δ ≈ 37/315.
        We use a Thwaites-like start: θ₀ = √(0.075 ν s₀ / Ue₀)
        with s₀ = ν / Ue₀.
        """
        if Ue0 <= 0:
            raise ValueError(f"Ue0 must be positive, got {Ue0}")
        s0 = nu / Ue0
        return math.sqrt(0.075 * nu * s0 / Ue0)

    def stagnation_theta(self, nu: float, K: float) -> float:
        """
        Pohlhausen stagnation patching: θ_stag = √(0.0770 · ν / K).

        At the stagnation point Λ_stag = 7.052, giving θ/δ and H from
        the polynomial, then the L'Hôpital limit of the momentum integral
        yields C = 0.0770.

        Args:
            nu: Kinematic viscosity [m²/s].
            K: Velocity gradient dUe/ds at stagnation [1/s].

        Returns:
            Stagnation momentum thickness θ_stag [m].
        """
        self._validate_stagnation_args(nu, K)
        return math.sqrt(0.0770 * nu / K)

    # ------------------------------------------------------------------
    # Pohlhausen integral relations  (functions of Λ)
    # ------------------------------------------------------------------

    @staticmethod
    def _delta_star_delta(Lambda: float) -> float:
        """δ*/δ = 3/10 − Λ/120."""
        return 3.0 / 10.0 - Lambda / 120.0

    @staticmethod
    def _theta_delta(Lambda: float) -> float:
        """θ/δ = 37/315 − Λ/945 − Λ²/9072."""
        return 37.0 / 315.0 - Lambda / 945.0 - Lambda ** 2 / 9072.0

    # ------------------------------------------------------------------
    # Post-processing: velocity field reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _H_to_Lambda(H: float) -> float:
        """Invert H = G(Λ)/Φ(Λ) to recover Λ via the quadratic formula.

        The equation rearranges to:
            (H/9072)Λ² + (H/945 − 1/120)Λ + (3/10 − 37H/315) = 0

        Returns the root in [−12, 12].
        """
        a = H / 9072.0
        b = H / 945.0 - 1.0 / 120.0
        c = 3.0 / 10.0 - 37.0 * H / 315.0

        disc = b * b - 4.0 * a * c
        if disc < 0:
            # Clamp to edge of valid range
            return -12.0 if H > 3.5 else 12.0

        sqrt_disc = math.sqrt(disc)
        r1 = (-b + sqrt_disc) / (2.0 * a)
        r2 = (-b - sqrt_disc) / (2.0 * a)

        # Pick root in valid range; prefer the one closest to zero
        for root in sorted([r1, r2], key=abs):
            if -12.0 <= root <= 12.0:
                return root

        # Both outside — clamp the closer one
        return max(-12.0, min(12.0, r1 if abs(r1) < abs(r2) else r2))

    @staticmethod
    def _profile_function(
        eta: NDArray[np.float64],
        Lambda: float,
    ) -> NDArray[np.float64]:
        """Evaluate the Pohlhausen profile g(η; Λ) on η ∈ [0, 1].

        g = 2η − 2η³ + η⁴ + (Λ/6)·η·(1 − η)³

        Args:
            eta: Normalised wall-normal coordinate, shape (Ny,).
            Lambda: Pohlhausen pressure-gradient parameter.

        Returns:
            u/Ue profile values, shape (Ny,).
        """
        e = np.clip(eta, 0.0, 1.0)
        return (
            2.0 * e - 2.0 * e**3 + e**4
            + (Lambda / 6.0) * e * (1.0 - e) ** 3
        )

    def compute_delta(self, theta: float, H: float) -> float:
        """
        Pohlhausen boundary layer thickness δ = θ / Φ(Λ).

        Args:
            theta: Momentum thickness θ [m].
            H: Shape factor δ*/θ at this station.

        Returns:
            Boundary layer thickness δ [m].
        """
        Lambda = self._H_to_Lambda(H)
        Phi = self._theta_delta(Lambda)
        if abs(Phi) < 1e-15:
            Phi = 1e-15
        return theta / Phi

    def reconstruct_velocity(
        self,
        y: NDArray[np.float64],
        theta: float,
        H: float,
        Ue: float,
    ) -> NDArray[np.float64]:
        """
        Reconstruct u(y) from the Pohlhausen polynomial profile.

        u(y) = Ue · g(y/δ; Λ)  for y ≤ δ,  u = Ue for y > δ.

        Args:
            y: Wall-normal coordinates [m], shape (Ny,).
            theta: Momentum thickness θ [m].
            H: Shape factor δ*/θ.
            Ue: Edge velocity [m/s].

        Returns:
            Velocity u(y) [m/s], shape (Ny,).
        """
        Lambda = self._H_to_Lambda(H)
        delta = self.compute_delta(theta, H)
        y_arr = np.asarray(y, dtype=np.float64)
        eta = y_arr / delta
        # Inside BL: use profile; outside: clamp to Ue
        u = Ue * self._profile_function(eta, Lambda)
        u = np.where(y_arr > delta, Ue, u)
        return u
