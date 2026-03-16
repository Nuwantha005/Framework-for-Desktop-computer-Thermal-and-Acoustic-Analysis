"""
Falkner-Skan wedge-flow velocity profile family.

The Falkner-Skan similarity solutions describe laminar boundary layers
in wedge flows where the edge velocity varies as a power law:

    Ue(x) ∝ x^m

The similarity parameter is β = 2m / (m + 1).  The Falkner-Skan ODE:

    f''' + f f'' + β (1 − f'²) = 0

must be solved numerically for each β.  Here we use tabulated/correlated
results (White, *Viscous Fluid Flow* 3rd ed., Table 4-3) to provide H
and cf/2 as functions of β.

The pressure-gradient parameter λ = θ²/ν · dUe/ds maps to β through
the similarity relationships.

References
----------
* White, *Viscous Fluid Flow* (3rd ed.), §4-3.
* Schlichting, *Boundary-Layer Theory* (8th ed.), §7.4.
* Katz & Plotkin, *Low-Speed Aerodynamics* (2nd ed.), §14.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .base import VelocityProfile, ProfileClosureData
from .tables import falkner_skan_table


# Tabulated Falkner-Skan results (β → H, S)
# S = cf/2 × Re_θ is the Thwaites-style shear parameter derived from
# the exact FS similarity solutions: S(β) = f''(0) × I_θ(β).
# Source: Cebeci & Bradshaw Table 4.1, White Table 4-3.
_FS_TABLE_BETA = np.array([
    -0.1988, -0.18, -0.12, -0.06,  0.0,
     0.10,    0.20,  0.30,  0.50,  1.00,  2.00,
])
_FS_TABLE_H = np.array([
    3.49, 3.38, 3.05, 2.80, 2.591,
    2.38, 2.22, 2.09, 1.89, 1.57, 1.33,
])
# S = cf/2 × Re_θ for each β  (S=0 at separation, 0.2205 for Blasius)
_FS_TABLE_S = np.array([
    0.000, 0.019, 0.076, 0.139, 0.2205,
    0.272, 0.306, 0.328, 0.354, 0.381, 0.395,
])


class FalknerSkanProfile(VelocityProfile):
    """
    Falkner-Skan similarity profile family.

    Provides H and cf/2 by interpolating tabulated solutions of the
    Falkner-Skan ODE indexed by the pressure-gradient parameter β.

    The Thwaites-style λ = θ²/ν · dUe/ds is mapped to β using the
    approximate relation β ≈ 2λ for small λ (exact for similarity flows).
    """

    @property
    def name(self) -> str:
        return "Falkner-Skan"

    def compute_closure(
        self,
        Re_theta: float,
        lambda_param: float = 0.0,
    ) -> ProfileClosureData:
        """
        Evaluate Falkner-Skan closure via table interpolation.

        Args:
            Re_theta: Momentum-thickness Reynolds number.
            lambda_param: Thwaites-style pressure parameter θ²/ν · dUe/ds.

        Returns:
            ProfileClosureData with H and cf/2 interpolated from the
            Falkner-Skan solution table.
        """
        if Re_theta <= 0:
            raise ValueError(f"Re_theta must be positive, got {Re_theta}")

        # Map Thwaites λ to Falkner-Skan β.
        # For FS similarity flows: λ = θ²/ν · dUe/ds and Ue = C x^m
        # ⇒ λ ≈ m · f''(0)² (exact), but β = 2m/(m+1).
        # Approximate: β ≈ 2λ is adequate for small λ; clamp to table range.
        beta = 2.0 * lambda_param
        beta = max(_FS_TABLE_BETA[0], min(_FS_TABLE_BETA[-1], beta))

        H = float(np.interp(beta, _FS_TABLE_BETA, _FS_TABLE_H))
        S = float(np.interp(beta, _FS_TABLE_BETA, _FS_TABLE_S))

        # cf/2 = S(β) / Re_θ  (laminar scaling, consistent with Thwaites)
        cf_2 = S / Re_theta if Re_theta > 0 else 0.0

        return ProfileClosureData(H=H, cf_2=cf_2)

    def initial_theta(self, nu: float, Ue0: float) -> float:
        """
        Starting θ from Blasius-like estimate (β = 0 member of family).

        Args:
            nu: Kinematic viscosity [m²/s].
            Ue0: Edge velocity at starting station [m/s].
        """
        if Ue0 <= 0:
            raise ValueError(f"Ue0 must be positive, got {Ue0}")
        s0 = nu / Ue0
        return 0.664 * math.sqrt(nu * s0 / Ue0)

    def stagnation_theta(self, nu: float, K: float) -> float:
        """
        Hiemenz (β = 1) stagnation patching: θ_stag = √(0.08547 · ν / K).

        At a 2-D stagnation point the flow is the Hiemenz solution
        (Falkner-Skan with β = 1, m = 1).  Substituting Ue = K·s into
        the momentum integral with the exact Hiemenz closure (H = 2.216,
        f''(0) = 1.23259) and applying L'Hôpital's rule gives
        C = 2·f''(0)·(θ/δ)² / [shape-integral] = 0.08547.

        Args:
            nu: Kinematic viscosity [m²/s].
            K: Velocity gradient dUe/ds at stagnation [1/s].

        Returns:
            Stagnation momentum thickness θ_stag [m].
        """
        self._validate_stagnation_args(nu, K)
        return math.sqrt(0.08547 * nu / K)

    # ------------------------------------------------------------------
    # Post-processing: velocity field reconstruction
    # ------------------------------------------------------------------

    def compute_delta(self, theta: float, H: float) -> float:
        """
        Falkner-Skan δ₉₉ from θ and H.

        Inverts H → β via the tabulated ODE solutions, then
        L = θ / I₂(β),  δ₉₉ = η₉₉(β) · L.

        Args:
            theta: Momentum thickness θ [m].
            H: Shape factor δ*/θ.

        Returns:
            Boundary layer thickness δ₉₉ [m].
        """
        tbl = falkner_skan_table()
        try:
            c = tbl.constants_from_H(H)
        except ValueError:
            # H outside tabulated range — fall back to Blasius-like estimate
            c = tbl.constants(0.0)
        L = theta / c.I_2
        return c.eta_99 * L

    def reconstruct_velocity(
        self,
        y: NDArray[np.float64],
        theta: float,
        H: float,
        Ue: float,
    ) -> NDArray[np.float64]:
        """
        Reconstruct u(y) from the Falkner-Skan similarity profile.

        Inverts H → β, then u(y) = Ue · f'_β(y/L)  where L = θ/I₂(β).

        Args:
            y: Wall-normal coordinates [m], shape (Ny,).
            theta: Momentum thickness θ [m].
            H: Shape factor δ*/θ.
            Ue: Edge velocity [m/s].

        Returns:
            Velocity u(y) [m/s], shape (Ny,).
        """
        tbl = falkner_skan_table()
        try:
            c = tbl.constants_from_H(H)
        except ValueError:
            c = tbl.constants(0.0)
        L = theta / c.I_2
        eta = np.asarray(y, dtype=np.float64) / L
        return Ue * tbl.fprime(c.beta, eta)
