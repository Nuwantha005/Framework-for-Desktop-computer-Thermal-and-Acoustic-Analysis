"""
Linear-strength doublet panel influence coefficient computations for 2D flow.

Implements potential and velocity influences for linearly varying doublet
panels following Katz & Plotkin, §10.3.2 and §11.5.1.

The doublet strength varies linearly across a panel of length S:

    μ(ξ) = μ_A (1 − ξ/S) + μ_B (ξ/S)

where ξ is the panel-local coordinate (0 at node A, S at node B).

The potential influence is decomposed as constant + linear (K&P Eqs. 11.111–11.115):

    Φ = Φ^a · μ_A  +  Φ^b · μ_B

For the Dirichlet formulation (§11.5.1), only the potential influences matter.
The velocity influences are needed for off-body field evaluation.
"""

import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from core.geometry.mesh import Mesh


# ── Single-panel potential influence ────────────────────────────────────


def compute_linear_doublet_potential_influence(
    point: NDArray,
    panel_start: NDArray,
    panel_end: NDArray,
) -> Tuple[float, float]:
    """
    Potential influence of a linear-strength doublet panel at a field point.

    Returns (Φ_a, Φ_b) such that Φ = μ_A · Φ_a + μ_B · Φ_b.

    Based on K&P Eqs. 11.114 and 11.115 in panel-local coordinates:

        Φ^a = -(1/2π) [Δθ − (x·Δθ + z/2·ln(r₂²/r₁²)) / S]
        Φ^b = -(1/2πS) [x·Δθ + z/2·ln(r₂²/r₁²)]

    Args:
        point: (x, y) field point in global coordinates.
        panel_start: (x, y) panel start node (node A).
        panel_end: (x, y) panel end node (node B).

    Returns:
        (Phi_a, Phi_b) potential influence coefficients.
    """
    x, z = point[0], point[1]
    x1, z1 = panel_start[0], panel_start[1]
    x2, z2 = panel_end[0], panel_end[1]

    dx_panel = x2 - x1
    dz_panel = z2 - z1
    S = np.sqrt(dx_panel**2 + dz_panel**2)

    if S < 1e-14:
        return 0.0, 0.0

    # Panel unit vectors
    tx = dx_panel / S
    tz = dz_panel / S
    nx = -tz  # normal (90° CCW rotation of tangent)
    nz = tx

    # Transform to panel-local coords (origin at node A, x along panel)
    rel_x = x - x1
    rel_z = z - z1
    x_loc = rel_x * tx + rel_z * tz
    z_loc = rel_x * nx + rel_z * nz

    # --- Self-influence (point on panel interior) ---
    if abs(z_loc) < 1e-12 and 0.0 < x_loc < S:
        # From the interior side (z → 0⁻), K&P Eq. 11.111a:
        #   Φ(x, 0⁻) = +(1/2)[μ₀ + μ₁·x]
        # For unit μ_A (μ_B = 0):  Φ^a = +(1/2)(1 − x_loc/S)
        # For unit μ_B (μ_A = 0):  Φ^b = +(1/2)(x_loc/S)
        Phi_a = 0.5 * (1.0 - x_loc / S)
        Phi_b = 0.5 * (x_loc / S)
        return Phi_a, Phi_b

    # --- General case ---
    # Distances from panel endpoints
    r1_sq = x_loc**2 + z_loc**2
    r2_sq = (x_loc - S)**2 + z_loc**2
    r1_sq = max(r1_sq, 1e-24)
    r2_sq = max(r2_sq, 1e-24)

    # Angles subtended
    theta1 = np.arctan2(z_loc, x_loc)
    theta2 = np.arctan2(z_loc, x_loc - S)
    d_theta = theta2 - theta1

    # Log ratio
    log_ratio = np.log(r2_sq / r1_sq)  # ln(r₂²/r₁²)

    # Common term: x·Δθ + (z/2)·ln(r₂²/r₁²)
    term = x_loc * d_theta + 0.5 * z_loc * log_ratio

    inv_2pi = 1.0 / (2.0 * np.pi)

    # K&P Eq. 11.114: Φ^a = -(1/2π)[Δθ − term/S]
    Phi_a = -inv_2pi * (d_theta - term / S)

    # K&P Eq. 11.115: Φ^b = -(1/2πS) · term
    Phi_b = -inv_2pi * term / S

    return Phi_a, Phi_b


# ── Single-panel source potential influence (linear strength) ───────────


def compute_linear_source_potential_influence(
    point: NDArray,
    panel_start: NDArray,
    panel_end: NDArray,
) -> Tuple[float, float]:
    """
    Potential influence of a linear-strength source panel at a field point.

    Returns (B_a, B_b) such that Φ_source = σ_A · B_a + σ_B · B_b.

    Based on K&P Eqs. 11.108–11.110 decomposed into node contributions.

    The constant-source potential (K&P Eq. 10.22) is:
        Φ₀ = (σ₀/4π)[(x-x₁)ln r₁² − (x-x₂)ln r₂² + 2z·Δθ]

    The linear-source addition (K&P Eq. 10.47) is:
        Φ₁ = (σ₁/4π)[½(x²−z²)(ln r₁²−ln r₂²) + 2xz·Δθ − x·S]
           + extra terms with endpoint evaluations

    Combined and split into node-A and node-B contributions.

    Args:
        point: (x, y) field point in global coordinates.
        panel_start: (x, y) panel start node (node A).
        panel_end: (x, y) panel end node (node B).

    Returns:
        (B_a, B_b) source potential influence coefficients.
    """
    x, z = point[0], point[1]
    x1, z1 = panel_start[0], panel_start[1]
    x2, z2 = panel_end[0], panel_end[1]

    dx_panel = x2 - x1
    dz_panel = z2 - z1
    S = np.sqrt(dx_panel**2 + dz_panel**2)

    if S < 1e-14:
        return 0.0, 0.0

    tx = dx_panel / S
    tz = dz_panel / S
    nx = -tz
    nz = tx

    rel_x = x - x1
    rel_z = z - z1
    x_loc = rel_x * tx + rel_z * tz
    z_loc = rel_x * nx + rel_z * nz

    # --- Self-influence (point on panel, at midpoint x_loc = S/2, z_loc = 0) ---
    if abs(z_loc) < 1e-12 and 0.0 < x_loc < S:
        # On-panel limit (z → 0) for source potential.
        # Constant part: Φ₀ = (1/4π)[x·ln(x²) − (x−S)·ln((x−S)²)]
        # Linear part (integral of ξ·ln((x−ξ)²) from 0 to S):
        #   Φ₁ = (1/4π)[½x²·ln(x²) + ½(S²−x²)·ln((x−S)²) − xS − S²/2]
        xm = x_loc
        xmS = x_loc - S

        ln_r1_sq = np.log(max(xm**2, 1e-30))
        ln_r2_sq = np.log(max(xmS**2, 1e-30))

        inv_4pi = 1.0 / (4.0 * np.pi)

        # Constant source potential on panel (z=0): full integral including -2S constant
        phi_const = inv_4pi * (xm * ln_r1_sq - xmS * ln_r2_sq - 2.0 * S)

        # Linear source potential on panel (z=0): exact integrated form
        phi_linear = inv_4pi * (
            0.5 * xm**2 * ln_r1_sq
            + 0.5 * (S**2 - xm**2) * ln_r2_sq
            - xm * S - S**2 / 2.0
        )

        # Total: Φ = σ_A·(phi_const − phi_linear/S) + σ_B·(phi_linear/S)
        B_a = phi_const - phi_linear / S
        B_b = phi_linear / S
        return B_a, B_b

    # --- General case ---
    r1_sq = x_loc**2 + z_loc**2
    r2_sq = (x_loc - S)**2 + z_loc**2
    r1_sq = max(r1_sq, 1e-24)
    r2_sq = max(r2_sq, 1e-24)

    ln_r1_sq = np.log(r1_sq)
    ln_r2_sq = np.log(r2_sq)

    theta1 = np.arctan2(z_loc, x_loc)
    theta2 = np.arctan2(z_loc, x_loc - S)
    d_theta = theta2 - theta1

    inv_4pi = 1.0 / (4.0 * np.pi)

    # Constant-source potential (full integral of ln((x-ξ)²+z²) from 0 to S):
    #   Φ₀ = (1/4π)[x·ln(r₁²) − (x−S)·ln(r₂²) + 2z·Δθ − 2S]
    phi_const = inv_4pi * (
        x_loc * ln_r1_sq - (x_loc - S) * ln_r2_sq + 2.0 * z_loc * d_theta - 2.0 * S
    )

    # Linear-source potential (integral of ξ·ln((x−ξ)²+z²) from 0 to S):
    #   Φ_lin = (1/4π)[ x·{x·ln(r₁²) − (x−S)·ln(r₂²) + 2z·Δθ}
    #                    − ½{r₁²·ln(r₁²) − r₂²·ln(r₂²)} − xS − S²/2 ]
    # Derived from integration by parts of ∫₀ˢ ξ·ln((x−ξ)²+z²) dξ.
    phi_const_raw = x_loc * ln_r1_sq - (x_loc - S) * ln_r2_sq + 2.0 * z_loc * d_theta

    phi_lin = inv_4pi * (
        x_loc * phi_const_raw
        - 0.5 * (r1_sq * ln_r1_sq - r2_sq * ln_r2_sq)
        - x_loc * S - S**2 / 2.0
    )

    # Total: Φ = σ₀·phi_const + σ₁·phi_lin
    # where σ₀ = σ_A, σ₁ = (σ_B − σ_A)/S
    # Rearrange: Φ = σ_A·(phi_const − phi_lin/S) + σ_B·(phi_lin/S)
    B_a = phi_const - phi_lin / S
    B_b = phi_lin / S

    return B_a, B_b


# ── Influence matrix assembly ───────────────────────────────────────────


def compute_linear_doublet_influence_matrix(mesh: Mesh) -> NDArray:
    """
    Assemble the doublet potential influence matrix C for the Dirichlet BC.

    C[i, k] is the total potential at collocation point i due to unit doublet
    strength at node k. For a linearly varying doublet across panel j connecting
    nodes n1 and n2, the contributions from Φ^a and Φ^b are accumulated into
    columns n1 and n2 respectively (K&P Eq. 11.117).

    Shape: (N_panels, N_nodes).

    Args:
        mesh: 2D panel mesh (closed body: N_panels == N_nodes).

    Returns:
        (N, N) doublet potential influence matrix.
    """
    n_panels = mesh.num_panels
    n_nodes = mesh.num_nodes
    centers = mesh.centers[:, :2]
    nodes = mesh.nodes[:, :2]

    C = np.zeros((n_panels, n_nodes), dtype=np.float64)

    for i in range(n_panels):
        pt = centers[i]

        for j in range(n_panels):
            n1 = mesh.panels[j, 0]
            n2 = mesh.panels[j, 1]
            p_start = nodes[n1]
            p_end = nodes[n2]

            if i == j:
                # Analytical self-influence at panel midpoint from interior
                S = mesh.areas[j]
                x_mid = S / 2.0
                C[i, n1] += 0.5 * (1.0 - x_mid / S)  # = 0.25
                C[i, n2] += 0.5 * (x_mid / S)          # = 0.25
            else:
                Phi_a, Phi_b = compute_linear_doublet_potential_influence(
                    point=pt, panel_start=p_start, panel_end=p_end,
                )
                C[i, n1] += Phi_a
                C[i, n2] += Phi_b

    return C


def compute_linear_source_potential_matrix(mesh: Mesh) -> NDArray:
    """
    Assemble the source potential influence matrix B for the Dirichlet BC.

    B[i, k] is the total source-induced potential at collocation point i
    due to unit source strength at node k. Linear source panels connecting
    nodes n1 and n2 contribute B_a to column n1 and B_b to column n2.

    Shape: (N_panels, N_nodes).

    Args:
        mesh: 2D panel mesh.

    Returns:
        (N, N) source potential influence matrix.
    """
    n_panels = mesh.num_panels
    n_nodes = mesh.num_nodes
    centers = mesh.centers[:, :2]
    nodes = mesh.nodes[:, :2]

    B = np.zeros((n_panels, n_nodes), dtype=np.float64)

    for i in range(n_panels):
        pt = centers[i]

        for j in range(n_panels):
            n1 = mesh.panels[j, 0]
            n2 = mesh.panels[j, 1]
            p_start = nodes[n1]
            p_end = nodes[n2]

            B_a, B_b = compute_linear_source_potential_influence(
                point=pt, panel_start=p_start, panel_end=p_end,
            )
            B[i, n1] += B_a
            B[i, n2] += B_b

    return B


# ── Velocity influence for off-body field evaluation ────────────────────


def compute_linear_doublet_velocity_influence(
    point: NDArray,
    panel_start: NDArray,
    panel_end: NDArray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Velocity influence of a linear-strength doublet panel at a field point.

    Returns ((u_a, w_a), (u_b, w_b)) such that:
        V = μ_A·(u_a, w_a) + μ_B·(u_b, w_b)

    A linear doublet μ₀ + μ₁·ξ decomposes into:
    - Constant part μ₀: two opposite point vortices at endpoints (K&P §10.2.2)
    - Linear part μ₁·ξ: constant vortex sheet γ = −μ₁ + two point vortices
      (K&P Eq. 10.61)

    Implementation uses direct differentiation of the potential (Eqs. 10.62–10.63)
    plus the constant-doublet velocity (Eqs. 10.29–10.30).

    Args:
        point: (x, y) field point in global coordinates.
        panel_start: (x, y) panel start node (node A).
        panel_end: (x, y) panel end node (node B).

    Returns:
        ((u_a, w_a), (u_b, w_b)) velocity influence tuples.
    """
    x, z = point[0], point[1]
    x1, z1 = panel_start[0], panel_start[1]
    x2, z2 = panel_end[0], panel_end[1]

    dx_panel = x2 - x1
    dz_panel = z2 - z1
    S = np.sqrt(dx_panel**2 + dz_panel**2)

    if S < 1e-14:
        return (0.0, 0.0), (0.0, 0.0)

    tx = dx_panel / S
    tz = dz_panel / S
    nx_hat = -tz
    nz_hat = tx

    rel_x = x - x1
    rel_z = z - z1
    x_loc = rel_x * tx + rel_z * tz
    z_loc = rel_x * nx_hat + rel_z * nz_hat

    # On-panel guard
    if abs(z_loc) < 1e-10 and -1e-10 < x_loc < S + 1e-10:
        return (0.0, 0.0), (0.0, 0.0)

    r1_sq = x_loc**2 + z_loc**2
    r2_sq = (x_loc - S)**2 + z_loc**2
    r1_sq = max(r1_sq, 1e-24)
    r2_sq = max(r2_sq, 1e-24)

    theta1 = np.arctan2(z_loc, x_loc)
    theta2 = np.arctan2(z_loc, x_loc - S)
    d_theta = theta2 - theta1
    log_ratio = np.log(r2_sq / r1_sq)

    inv_2pi = 1.0 / (2.0 * np.pi)

    # --- Constant doublet velocity (for μ₀ = 1, K&P Eqs. 10.29/10.30) ---
    u_const = -inv_2pi * (z_loc / r1_sq - z_loc / r2_sq)
    w_const = inv_2pi * (x_loc / r1_sq - (x_loc - S) / r2_sq)

    # --- Linear doublet velocity addition (for μ₁ = 1, K&P Eqs. 10.62/10.63) ---
    # u_lin = −(1/2π)Δθ + endpoint vortex terms
    u_lin_loc = -inv_2pi * d_theta + inv_2pi * (
        S * z_loc / r2_sq  # x₂·z/r₂² − x₁·z/r₁² with x₁=0, x₂=S
    )
    # w_lin = −(1/4π)ln(r₂²/r₁²) + endpoint terms
    w_lin_loc = -0.25 / np.pi * log_ratio + inv_2pi * (
        -S * (x_loc - S) / r2_sq  # x₁(x−x₁)/r₁² − x₂(x−x₂)/r₂²
    )

    # Total velocity in local coords for unit μ₀ and unit μ₁:
    # V = μ₀·(u_const, w_const) + μ₁·(u_lin, w_lin)
    # μ₀ = μ_A, μ₁ = (μ_B − μ_A)/S
    # So: V = μ_A·(u_const − u_lin/S, w_const − w_lin/S) + μ_B·(u_lin/S, w_lin/S)

    u_a_loc = u_const - u_lin_loc / S
    w_a_loc = w_const - w_lin_loc / S
    u_b_loc = u_lin_loc / S
    w_b_loc = w_lin_loc / S

    # Rotate back to global
    u_a = u_a_loc * tx - w_a_loc * tz
    w_a = u_a_loc * tz + w_a_loc * tx
    u_b = u_b_loc * tx - w_b_loc * tz
    w_b = u_b_loc * tz + w_b_loc * tx

    return (u_a, w_a), (u_b, w_b)


def compute_linear_doublet_velocity_field(
    points: NDArray,
    mesh: Mesh,
    mu: NDArray,
) -> NDArray:
    """
    Compute doublet-induced velocity field at given points.

    Args:
        points: (M, 2) or (M, 3) coordinates.
        mesh: Panel mesh.
        mu: (N_nodes,) doublet strengths at nodes.

    Returns:
        (M, 2) velocity vectors (Vx, Vy) from doublet contribution only.
    """
    n_points = points.shape[0]
    n_panels = mesh.num_panels
    nodes = mesh.nodes[:, :2]

    V = np.zeros((n_points, 2), dtype=np.float64)

    for j in range(n_panels):
        n1 = mesh.panels[j, 0]
        n2 = mesh.panels[j, 1]
        p_start = nodes[n1]
        p_end = nodes[n2]
        mu_a = mu[n1]
        mu_b = mu[n2]

        for k in range(n_points):
            (u_a, w_a), (u_b, w_b) = compute_linear_doublet_velocity_influence(
                point=points[k, :2], panel_start=p_start, panel_end=p_end,
            )
            V[k, 0] += mu_a * u_a + mu_b * u_b
            V[k, 1] += mu_a * w_a + mu_b * w_b

    return V
