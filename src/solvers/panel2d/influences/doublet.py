"""
Constant-strength doublet panel influence coefficient computations for 2D flow.

Implements the Dirichlet (internal potential) boundary condition formulation
from Katz & Plotkin, *Low-Speed Aerodynamics*, Sections 10.2.2 and 11.3.

The doublet potential influence at a field point P is (K&P Eq. 10.28):

    Φ = (-μ / 2π) [arctan(z/(x-x₂)) - arctan(z/(x-x₁))]
      = (-μ / 2π) Δθ

where Δθ is the angle subtended by the panel at P.

For the Morino (source + doublet) formulation, panel sources are prescribed
(σ_j = n_j · V∞) and only doublet strengths μ_j are solved from the
Dirichlet internal-potential condition.
"""

import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from core.geometry.mesh import Mesh


def compute_doublet_potential_influence(
    point: NDArray,
    panel_start: NDArray,
    panel_end: NDArray,
) -> float:
    """
    Compute potential influence of a unit-strength constant doublet panel at a point.

    Based on K&P Eq. 10.28:
        Φ = (-μ / 2π) [arctan(z/(x - x₂)) - arctan(z/(x - x₁))]

    Returns the coefficient c such that Φ = μ * c (i.e. c = -Δθ / 2π).

    Args:
        point: (x, z) field point coordinates (global frame).
        panel_start: (x₁, z₁) panel start node.
        panel_end: (x₂, z₂) panel end node.

    Returns:
        Potential influence coefficient c_{ij} (multiply by μ to get Φ).
    """
    x, z = point[0], point[1]
    x1, z1 = panel_start[0], panel_start[1]
    x2, z2 = panel_end[0], panel_end[1]

    # Panel geometry
    dx_panel = x2 - x1
    dz_panel = z2 - z1
    panel_length = np.sqrt(dx_panel**2 + dz_panel**2)

    if panel_length < 1e-14:
        return 0.0

    # Unit vectors along and normal to panel
    tx = dx_panel / panel_length
    tz = dz_panel / panel_length
    nx = -tz  # normal (rotate tangent 90° CCW)
    nz = tx

    # Transform point to panel-local coordinates
    # Origin at panel_start, x-axis along panel
    rel_x = x - x1
    rel_z = z - z1
    x_loc = rel_x * tx + rel_z * tz   # along panel
    z_loc = rel_x * nx + rel_z * nz   # normal to panel

    S = panel_length

    # Guard against point on panel (z_loc ≈ 0 and 0 < x_loc < S)
    if abs(z_loc) < 1e-12 and 0.0 < x_loc < S:
        # Self-influence: on the panel interior from the inside (z → 0⁻)
        # Φ = +μ/2  →  c = +1/2  (from the interior side, K&P Eq. 10.31)
        return 0.5

    # Angles subtended (K&P Eq. 10.28)
    theta1 = np.arctan2(z_loc, x_loc)
    theta2 = np.arctan2(z_loc, x_loc - S)
    d_theta = theta2 - theta1

    # c = -Δθ / (2π)
    return -d_theta / (2.0 * np.pi)


def compute_doublet_influence_matrix(mesh: Mesh) -> NDArray:
    """
    Compute the doublet potential influence matrix C for the Dirichlet BC.

    C[i, j] is the potential at collocation point i due to a unit-strength
    constant doublet on panel j:
        C[i, j] = -(1/2π) Δθ_{ij}

    Self-influence (i == j): C[i, i] = 1/2 (K&P Eq. 11.69).

    Args:
        mesh: 2D panel mesh.

    Returns:
        (N, N) doublet potential influence matrix.
    """
    n_panels = mesh.num_panels
    centers = mesh.centers[:, :2]

    panel_start_idx = mesh.panels[:, 0]
    panel_end_idx = mesh.panels[:, 1]
    nodes_start = mesh.nodes[panel_start_idx, :2]
    nodes_end = mesh.nodes[panel_end_idx, :2]

    C = np.zeros((n_panels, n_panels), dtype=np.float64)

    for i in range(n_panels):
        for j in range(n_panels):
            if i == j:
                # Self-influence from interior collocation: K&P Eq. 11.69
                C[i, j] = 0.5
            else:
                C[i, j] = compute_doublet_potential_influence(
                    point=centers[i],
                    panel_start=nodes_start[j],
                    panel_end=nodes_end[j],
                )

    return C


def compute_source_potential_matrix(mesh: Mesh) -> NDArray:
    """
    Compute the source potential influence matrix B for the Dirichlet BC.

    B[i, j] is the potential at collocation point i due to a unit-strength
    constant source on panel j:
        Φ_source = (σ / 4π) * potential_coeff

    So B[i, j] = potential_coeff / (4π), i.e. Φ = σ_j * B[i, j].

    Self-influence (i == j): uses the analytical on-panel formula from K&P.

    Args:
        mesh: 2D panel mesh.

    Returns:
        (N, N) source potential influence matrix.
    """
    from .source import compute_source_potential_influence

    n_panels = mesh.num_panels
    centers = mesh.centers[:, :2]

    panel_start_idx = mesh.panels[:, 0]
    panel_end_idx = mesh.panels[:, 1]
    nodes_start = mesh.nodes[panel_start_idx, :2]
    nodes_end = mesh.nodes[panel_end_idx, :2]

    B = np.zeros((n_panels, n_panels), dtype=np.float64)

    for i in range(n_panels):
        for j in range(n_panels):
            if i == j:
                # Self-influence: K&P Eq. 10.22a
                # Φ = (σ/4π) * S * [ln(S/2)² - 2 + 2·0] at midpoint
                # But simpler: at midpoint of panel of length S,
                # the contribution is S * ln(S/2)² analytically
                # We store coeff/(4π) so that Φ = σ * B
                S = mesh.areas[j]
                self_coeff = S * np.log((S / 2.0) ** 2)
                B[i, j] = self_coeff / (4.0 * np.pi)
            else:
                raw_coeff = compute_source_potential_influence(
                    point=centers[i],
                    panel_start=nodes_start[j],
                    panel_end=nodes_end[j],
                )
                B[i, j] = raw_coeff / (4.0 * np.pi)

    return B


def compute_doublet_velocity_influence(
    point: NDArray,
    panel_start: NDArray,
    panel_end: NDArray,
) -> Tuple[float, float]:
    """
    Compute velocity influence of a unit-strength constant doublet panel at a field point.

    A constant doublet panel is equivalent to two opposite point vortices at the
    panel endpoints (K&P text after Eq. 10.28). The induced velocity is:

        u = (-μ/2π) [z/((x-x₁)² + z²) - z/((x-x₂)² + z²)]
        w = ( μ/2π) [(x-x₁)/((x-x₁)² + z²) - (x-x₂)/((x-x₂)² + z²)]

    Returns (u_coeff, w_coeff) such that (u, w) = μ * (u_coeff, w_coeff).

    Args:
        point: (x, z) field point coordinates (global frame).
        panel_start: (x₁, z₁) panel start node.
        panel_end: (x₂, z₂) panel end node.

    Returns:
        Tuple (u_coeff, w_coeff) velocity influence for unit μ.
    """
    x, z = point[0], point[1]
    x1, z1 = panel_start[0], panel_start[1]
    x2, z2 = panel_end[0], panel_end[1]

    # Panel geometry
    dx_panel = x2 - x1
    dz_panel = z2 - z1
    panel_length = np.sqrt(dx_panel**2 + dz_panel**2)

    if panel_length < 1e-14:
        return 0.0, 0.0

    # Transform to panel-local coordinates
    tx = dx_panel / panel_length
    tz = dz_panel / panel_length
    nx = -tz
    nz = tx

    rel_x = x - x1
    rel_z = z - z1
    x_loc = rel_x * tx + rel_z * tz
    z_loc = rel_x * nx + rel_z * nz

    S = panel_length

    # Distances squared from panel endpoints
    r1_sq = x_loc**2 + z_loc**2
    r2_sq = (x_loc - S)**2 + z_loc**2

    # Guard against singularity
    r1_sq = max(r1_sq, 1e-24)
    r2_sq = max(r2_sq, 1e-24)

    # K&P Eqs. 10.29, 10.30 in local coords
    inv_2pi = 1.0 / (2.0 * np.pi)
    u_loc = -inv_2pi * (z_loc / r1_sq - z_loc / r2_sq)
    w_loc = inv_2pi * (x_loc / r1_sq - (x_loc - S) / r2_sq)

    # Rotate back to global
    u_global = u_loc * tx - w_loc * tz
    w_global = u_loc * tz + w_loc * tx

    # Guard: if point is on/very near the panel, zero out
    if abs(z_loc) < 1e-10 and -1e-10 < x_loc < S + 1e-10:
        return 0.0, 0.0

    return u_global, w_global
