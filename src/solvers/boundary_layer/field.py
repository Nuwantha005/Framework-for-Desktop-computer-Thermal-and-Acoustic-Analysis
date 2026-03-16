"""
Reconstructed boundary layer velocity field from integral results.

After the Von Kármán solver produces integral quantities (θ, H, Ue) at each
arc-length station, the velocity profile can be reconstructed by evaluating
the assumed profile shape at wall-normal coordinates.  This module provides:

- :class:`BLFieldData` — container for the reconstructed 2-D velocity field.
- :func:`reconstruct_bl_field` — batch reconstruction from a solver result.

The reconstructed field is the velocity distribution *consistent with the
integral method's closure assumptions*.  It is not the true Navier-Stokes
field — any discrepancy with CFD is a measure of the integral method's
modelling error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import BoundaryLayerResult
from .profiles.base import VelocityProfile


@dataclass
class BLFieldData:
    """Reconstructed 2-D boundary layer velocity field.

    All per-station arrays share the first dimension *M* (number of valid
    stations).  The wall-normal resolution *Ny* is uniform across stations.

    Attributes:
        s: Arc-length stations where the field is valid [m], shape (M,).
        y: Wall-normal coordinates per station [m], shape (M, Ny).
            y[i, :] ranges from 0 to ``delta[i]`` (or ``delta[i] * extend``
            if an extension factor was applied).
        u: Tangential velocity field [m/s], shape (M, Ny).
            u[i, j] = Ue[i] · g(y[i,j] / δ[i]).
        delta: Boundary layer thickness per station [m], shape (M,).
            For finite-domain profiles this is the exact edge δ; for
            similarity profiles this is δ₉₉.
        Ue: Edge velocity at each station [m/s], shape (M,).
        theta: Momentum thickness at each station [m], shape (M,).
        H: Shape factor at each station, shape (M,).
        profile_name: Name of the profile used for reconstruction.
    """

    s: NDArray[np.float64]
    y: NDArray[np.float64]
    u: NDArray[np.float64]
    delta: NDArray[np.float64]
    Ue: NDArray[np.float64]
    theta: NDArray[np.float64]
    H: NDArray[np.float64]
    profile_name: str = ""


def reconstruct_bl_field(
    result: BoundaryLayerResult,
    profile: VelocityProfile,
    n_y: int = 80,
    extend: float = 1.0,
) -> BLFieldData:
    """Reconstruct the velocity field from an integral BL solution.

    At each valid station (non-NaN θ), the profile's ``compute_delta``
    and ``reconstruct_velocity`` methods are called to build the 2-D
    field u(s, y).

    Args:
        result: Integral solver output (θ, H, Ue, s arrays).
        profile: Velocity profile used for closure (must implement
            ``compute_delta`` and ``reconstruct_velocity``).
        n_y: Number of wall-normal grid points per station.
        extend: Extension factor for the y-domain.  1.0 means
            y ∈ [0, δ]; 1.2 means y ∈ [0, 1.2δ], useful for
            showing the blend into the free stream.

    Returns:
        :class:`BLFieldData` with the reconstructed field.

    Raises:
        NotImplementedError: If the profile does not support
            reconstruction.
    """
    # Identify valid stations (non-NaN theta)
    valid = ~np.isnan(result.theta)
    idx = np.where(valid)[0]

    if len(idx) == 0:
        return BLFieldData(
            s=np.empty(0),
            y=np.empty((0, n_y)),
            u=np.empty((0, n_y)),
            delta=np.empty(0),
            Ue=np.empty(0),
            theta=np.empty(0),
            H=np.empty(0),
            profile_name=profile.name,
        )

    M = len(idx)
    s_out = result.s[idx]
    theta_out = result.theta[idx]
    H_out = result.H[idx]
    Ue_out = result.Ue[idx]

    # Compute delta at each station
    delta_out = np.empty(M, dtype=np.float64)
    for i in range(M):
        delta_out[i] = profile.compute_delta(theta_out[i], H_out[i])

    # Build y grid and reconstruct velocity
    y_out = np.empty((M, n_y), dtype=np.float64)
    u_out = np.empty((M, n_y), dtype=np.float64)

    for i in range(M):
        y_max = delta_out[i] * extend
        y_out[i] = np.linspace(0.0, y_max, n_y)
        u_out[i] = profile.reconstruct_velocity(
            y_out[i], theta_out[i], H_out[i], Ue_out[i],
        )

    return BLFieldData(
        s=s_out,
        y=y_out,
        u=u_out,
        delta=delta_out,
        Ue=Ue_out,
        theta=theta_out,
        H=H_out,
        profile_name=profile.name,
    )
