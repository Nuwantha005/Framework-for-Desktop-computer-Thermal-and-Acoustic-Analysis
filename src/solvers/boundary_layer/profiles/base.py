"""
Abstract base class for boundary layer velocity profile parameterizations.

Each velocity profile provides closure relations for the Von Kármán
momentum integral equation by relating the shape factor H, skin friction
coefficient cf, and other integral quantities to the profile parameters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


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

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
