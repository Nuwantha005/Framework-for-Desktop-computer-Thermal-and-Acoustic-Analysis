"""
Laminar–turbulent transition prediction models.

Provides transition criteria that can be applied to a
:class:`BoundaryLayerResult` to estimate the transition location.
Both algebraic correlations and envelope methods are included.

Supported criteria
------------------
* **Michel's criterion** — empirical Re_θ vs Re_x correlation.
* **e^N (simplified)** — single-parameter amplification model using
  the Drela/Arnal correlation ñ ≈ 9 by default.

These are *post-processing* utilities: run the laminar BL solver first,
then apply a transition criterion to locate s_tr.

References
----------
* Michel, R., "Détermination du point de transition …", *ONERA Rpt 58*,
  1951.
* Arnal, D., "Transition prediction in transonic flow", *IUTAM Symp.*,
  1988.
* Drela, M., *Flight Vehicle Aerodynamics*, MIT Press, 2014, §4.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TransitionResult:
    """
    Result of a transition-location prediction.

    Attributes:
        transition_s: Arc-length location of predicted transition [m],
            or ``None`` if the BL remains laminar.
        transition_index: Index in the arc-length array closest to the
            transition point, or ``None``.
        criterion_name: Name of the criterion that triggered.
        Re_theta_tr: Re_θ at the transition location.
    """
    transition_s: Optional[float]
    transition_index: Optional[int]
    criterion_name: str
    Re_theta_tr: Optional[float] = None


# ---------------------------------------------------------------------------
# Michel's criterion
# ---------------------------------------------------------------------------

def michel_criterion(
    s: NDArray[np.float64],
    Ue: NDArray[np.float64],
    theta: NDArray[np.float64],
    nu: float,
) -> TransitionResult:
    """
    Predict transition using Michel's empirical correlation.

    Transition occurs where:

        Re_θ ≥ 1.174 · Re_x^0.46

    where Re_x = Ue(s) · s / ν  and  Re_θ = Ue(s) · θ(s) / ν.

    Args:
        s: Arc-length stations, shape (M,).
        Ue: Edge velocity, shape (M,).
        theta: Momentum thickness, shape (M,).
        nu: Kinematic viscosity [m²/s].

    Returns:
        TransitionResult with the first station exceeding the criterion.
    """
    Re_theta = np.abs(Ue) * theta / nu
    Re_x = np.abs(Ue) * s / nu

    # Michel critical Re_θ
    Re_theta_crit = 1.174 * np.power(np.maximum(Re_x, 1.0), 0.46)

    exceeded = np.where(Re_theta >= Re_theta_crit)[0]

    if len(exceeded) == 0:
        return TransitionResult(
            transition_s=None,
            transition_index=None,
            criterion_name="Michel",
        )

    idx = int(exceeded[0])
    return TransitionResult(
        transition_s=float(s[idx]),
        transition_index=idx,
        criterion_name="Michel",
        Re_theta_tr=float(Re_theta[idx]),
    )


# ---------------------------------------------------------------------------
# Simplified e^N criterion
# ---------------------------------------------------------------------------

def en_criterion(
    s: NDArray[np.float64],
    Ue: NDArray[np.float64],
    theta: NDArray[np.float64],
    nu: float,
    n_crit: float = 9.0,
) -> TransitionResult:
    """
    Simplified e^N transition prediction.

    Uses the Drela/Arnal single-parameter model where the amplification
    factor *n* grows from the instability point.  Transition when n ≥ n_crit.

    The instability point is estimated as Re_θ,crit ≈ 150 (Tollmien-
    Schlichting threshold).  Growth rate approximation:

        dn/ds ≈ (Re_θ − Re_θ,crit) / (Re_θ · θ)

    integrated with the trapezoidal rule.

    Args:
        s: Arc-length stations, shape (M,).
        Ue: Edge velocity, shape (M,).
        theta: Momentum thickness, shape (M,).
        nu: Kinematic viscosity [m²/s].
        n_crit: Critical amplification factor (default 9.0).

    Returns:
        TransitionResult.
    """
    RE_THETA_INSTABILITY = 150.0

    Re_theta = np.abs(Ue) * theta / nu
    n_factor = np.zeros_like(s)

    for i in range(1, len(s)):
        if Re_theta[i] <= RE_THETA_INSTABILITY:
            n_factor[i] = 0.0
            continue

        # Simple growth rate model
        growth = (Re_theta[i] - RE_THETA_INSTABILITY) / (Re_theta[i] * theta[i])
        ds = s[i] - s[i - 1]
        n_factor[i] = n_factor[i - 1] + growth * ds

        if n_factor[i] >= n_crit:
            return TransitionResult(
                transition_s=float(s[i]),
                transition_index=i,
                criterion_name=f"e^N (N_crit={n_crit})",
                Re_theta_tr=float(Re_theta[i]),
            )

    return TransitionResult(
        transition_s=None,
        transition_index=None,
        criterion_name=f"e^N (N_crit={n_crit})",
    )
