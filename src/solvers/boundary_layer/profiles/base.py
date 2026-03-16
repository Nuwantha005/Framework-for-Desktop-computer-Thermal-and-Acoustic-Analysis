"""
Abstract base class for boundary layer velocity profile parameterizations.

Each velocity profile provides closure relations for the Von Kármán
momentum integral equation by relating the shape factor H, skin friction
coefficient cf, and other integral quantities to the profile parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import math

import numpy as np
from numpy.typing import NDArray


@dataclass
class ProfileClosureData:
    """
    Closure quantities derived from a velocity profile evaluation.

    These values close the Von Kármán momentum integral ODE at a single
    station along the surface.

    Attributes:
        H: Shape factor δ*/θ.
        cf_2: Half skin friction coefficient τ_w / (ρ U_e²) = cf / 2.
        delta_star_ratio: Displacement thickness ratio δ*/δ (optional,
            profile-dependent).
        theta_ratio: Momentum thickness ratio θ/δ (optional).
    """
    H: float
    cf_2: float
    delta_star_ratio: float | None = None
    theta_ratio: float | None = None


class VelocityProfile(ABC):
    """
    Base class for BL velocity profile parameterizations.

    A velocity profile provides the closure relations needed to integrate
    the Von Kármán momentum integral equation:

        dθ/ds + (θ/Ue)(dUe/ds)(2 + H) = cf/2

    Subclasses implement specific profile families (Blasius, Pohlhausen,
    Falkner-Skan, power-law, Thwaites) that each define how H and cf/2
    depend on flow conditions (Re_theta, pressure gradient parameter, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable profile name for legends and reports."""

    @abstractmethod
    def compute_closure(
        self,
        Re_theta: float,
        lambda_param: float = 0.0,
    ) -> ProfileClosureData:
        """
        Evaluate closure relations at a single BL station.

        Args:
            Re_theta: Momentum-thickness Reynolds number Ue·θ/ν.
            lambda_param: Pressure-gradient parameter.  Definition is
                profile-dependent (e.g. Λ = θ²/ν · dUe/ds for Thwaites,
                Λ = δ²/ν · dUe/ds for Pohlhausen).  Profiles that do not
                use a pressure gradient parameter may ignore this argument.

        Returns:
            ProfileClosureData with H and cf/2 (and optional ratios).

        Raises:
            ValueError: If inputs are outside the profile's valid range.
        """

    @abstractmethod
    def initial_theta(self, nu: float, Ue0: float) -> float:
        """
        Provide a starting momentum thickness at or near the stagnation point.

        .. deprecated::
            Prefer :meth:`stagnation_theta` when K is available (Phase 3).

        Many integral methods require a non-zero initial θ to avoid the
        singularity where Ue → 0.  Each profile family has its own
        recommended starting procedure (e.g. Thwaites' closed-form, Blasius
        similarity).

        Args:
            nu: Kinematic viscosity [m²/s].
            Ue0: Edge velocity at the first non-singular station [m/s].

        Returns:
            Initial momentum thickness θ₀ [m].

        Raises:
            ValueError: If Ue0 <= 0.
        """

    def stagnation_theta(self, nu: float, K: float) -> float:
        """
        Analytical momentum thickness at the stagnation point.

        Near a 2-D stagnation point the edge velocity grows linearly:
        Ue(s) ≈ K·s.  Substituting into the momentum integral equation
        and applying L'Hôpital's rule as s → 0 yields a profile-specific
        constant C such that:

            θ_stag = √(C · ν / K)

        Subclasses that support stagnation patching override this method
        with the correct constant.  Profiles for which analytical patching
        is not meaningful (e.g. turbulent power-law) raise
        ``NotImplementedError``.

        Args:
            nu: Kinematic viscosity [m²/s].
            K: Velocity gradient dUe/ds at the stagnation point [1/s].
                Must be positive.

        Returns:
            Stagnation momentum thickness θ_stag [m].

        Raises:
            ValueError: If nu <= 0 or K <= 0.
            NotImplementedError: If the profile does not support patching.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement stagnation_theta."
        )

    @staticmethod
    def _validate_stagnation_args(nu: float, K: float) -> None:
        """Common validation for stagnation_theta arguments."""
        if nu <= 0:
            raise ValueError(f"Kinematic viscosity must be positive, got {nu}")
        if K <= 0:
            raise ValueError(f"Velocity gradient K must be positive, got {K}")

    # ------------------------------------------------------------------
    # Post-processing: velocity field reconstruction (Phase 4)
    # ------------------------------------------------------------------

    def compute_delta(self, theta: float, H: float) -> float:
        """
        Compute the boundary layer thickness δ (or δ₉₉) at a station.

        For finite-domain profiles (Pohlhausen, Power-Law), this is the
        exact edge thickness δ where u = Ue.  For similarity profiles
        (Blasius, Falkner-Skan), this is δ₉₉ where u/Ue = 0.99.

        Subclasses override this with profile-specific formulas using
        the known analytical θ/δ ratios.

        Args:
            theta: Momentum thickness θ [m].
            H: Shape factor δ*/θ at this station.

        Returns:
            Boundary layer thickness [m].

        Raises:
            NotImplementedError: If the profile has not implemented
                reconstruction.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compute_delta."
        )

    def reconstruct_velocity(
        self,
        y: NDArray[np.float64],
        theta: float,
        H: float,
        Ue: float,
    ) -> NDArray[np.float64]:
        """
        Reconstruct the velocity profile u(y) at a single BL station.

        Evaluates the assumed profile shape at wall-normal coordinates *y*,
        returning the tangential velocity.  For y > δ the velocity is
        clamped to Ue.

        Args:
            y: Wall-normal coordinates [m], shape (Ny,).
            theta: Momentum thickness θ [m] at this station.
            H: Shape factor δ*/θ at this station.
            Ue: Edge velocity [m/s] at this station.

        Returns:
            Velocity array u(y) [m/s], shape (Ny,).

        Raises:
            NotImplementedError: If the profile has not implemented
                reconstruction.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement reconstruct_velocity."
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
