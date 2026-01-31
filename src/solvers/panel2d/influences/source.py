"""
Source panel influence coefficient computations for 2D constant-strength panels.

These functions compute the geometric integrals for source panel methods
following the Katz & Plotkin formulation.
"""

import numpy as np
import math
from numpy.typing import NDArray
from typing import Tuple

from core.geometry.mesh import Mesh


def compute_source_influence_matrices(mesh: Mesh) -> Tuple[NDArray, NDArray]:
    """
    Compute normal (I) and tangential (J) influence coefficient matrices for source panels.
    
    Based on Katz & Plotkin "Low Speed Aerodynamics" formulation for
    constant-strength source panels on flat line segments.
    
    Args:
        mesh: 2D panel mesh
    
    Returns:
        Tuple of (I, J) matrices, both shape (N, N) where N is number of panels:
        - I: Normal influence coefficients
        - J: Tangential influence coefficients
    
    Note:
        Self-influence (i==j) is handled by returning 0.0. The diagonal
        of I is set to π during linear system assembly.
    """
    n_panels = len(mesh.panels)
    
    # Extract geometry
    centers = mesh.centers[:, :2]  # (N, 2) control points
    
    # Panel start nodes
    panel_start_indices = mesh.panels[:, 0]
    nodes_start = mesh.nodes[panel_start_indices, :2]  # (N, 2)
    
    # Panel lengths
    panel_lengths = mesh.areas  # (N,)
    
    # Panel orientation angles
    tangents = mesh.tangents[:, :2]  # (N, 2)
    phi = np.arctan2(tangents[:, 1], tangents[:, 0])  # (N,)
    phi = np.where(phi < 0, phi + 2*np.pi, phi)
    
    # Initialize matrices
    I = np.zeros((n_panels, n_panels))
    J = np.zeros((n_panels, n_panels))
    
    # Compute influence coefficients
    for i in range(n_panels):
        for j in range(n_panels):
            if i == j:
                # Self-influence is 0 for the geometric integral
                # (π added to diagonal during system assembly)
                continue
            
            I[i, j], J[i, j] = _compute_panel_influence(
                control_point=centers[i],
                panel_start=nodes_start[j],
                panel_length=panel_lengths[j],
                panel_angle=phi[j],
                control_angle=phi[i]
            )
    
    return I, J


def _compute_panel_influence(
    control_point: NDArray,
    panel_start: NDArray,
    panel_length: float,
    panel_angle: float,
    control_angle: float
) -> Tuple[float, float]:
    """
    Compute influence coefficients for a single panel-to-control-point pair.
    
    Args:
        control_point: (x, y) coordinates of control point
        panel_start: (x, y) coordinates of panel start node
        panel_length: Panel length S
        panel_angle: Panel orientation angle φ_j
        control_angle: Control point panel orientation φ_i
    
    Returns:
        Tuple of (I_ij, J_ij) - normal and tangential influence coefficients
    """
    # Relative position
    dx = control_point[0] - panel_start[0]
    dy = control_point[1] - panel_start[1]
    
    # Transform to panel local coordinates
    A = -dx * np.cos(panel_angle) - dy * np.sin(panel_angle)
    B = dx**2 + dy**2
    
    # Orientation differences
    Cn = np.sin(control_angle - panel_angle)
    Dn = -dx * np.sin(control_angle) + dy * np.cos(control_angle)
    Ct = -np.cos(control_angle - panel_angle)
    Dt = dx * np.cos(control_angle) + dy * np.sin(control_angle)
    
    # Distance from panel line
    E_sq = B - A**2
    if E_sq <= 0:  # Numerical stability
        E = 0.0
    else:
        E = np.sqrt(E_sq)
    
    if E < 1e-12:
        # Point is on the panel line (or very close)
        return 0.0, 0.0
    
    # Logarithmic term
    log_term = np.log((panel_length**2 + 2*A*panel_length + B) / B)
    
    # Arctangent terms
    atan_term = (math.atan2(panel_length + A, E) - math.atan2(A, E))
    
    # Normal influence coefficient
    I_ij = 0.5 * Cn * log_term + ((Dn - A*Cn) / E) * atan_term
    
    # Tangential influence coefficient
    J_ij = 0.5 * Ct * log_term + ((Dt - A*Ct) / E) * atan_term
    
    return I_ij, J_ij


def compute_source_velocity_influence(
    point: NDArray,
    panel_start: NDArray,
    panel_length: float,
    panel_angle: float
) -> Tuple[float, float]:
    """
    Compute velocity influence of a source panel at an arbitrary field point.
    
    Returns the velocity coefficients (Mx, My) such that:
        u_induced = sigma * Mx / (2π)
        v_induced = sigma * My / (2π)
    
    where sigma is the source strength.
    
    Args:
        point: (x, y) field point coordinates
        panel_start: (x, y) panel start node
        panel_length: Panel length S
        panel_angle: Panel orientation φ
    
    Returns:
        Tuple of (Mx, My) velocity influence coefficients
    """
    # Relative position
    dx = point[0] - panel_start[0]
    dy = point[1] - panel_start[1]
    
    cos_phi = np.cos(panel_angle)
    sin_phi = np.sin(panel_angle)
    
    # Transform to panel local coordinates
    A = -dx * cos_phi - dy * sin_phi
    B = np.maximum(dx**2 + dy**2, 1e-12)
    
    # Distance from panel line
    E_sq = np.maximum(B - A**2, 0.0)
    E = np.sqrt(E_sq)
    
    # Influence coefficients in x, y directions
    Cx = -cos_phi
    Dx = dx
    Cy = -sin_phi
    Dy = dy
    
    # Logarithmic term
    log_term = np.log(np.maximum((panel_length**2 + 2*A*panel_length + B) / B, 1e-12))
    
    # Arctangent term
    E_safe = np.where(E < 1e-12, 1.0, E)
    atan_term = np.arctan2(panel_length + A, E_safe) - np.arctan2(A, E_safe)
    
    # Velocity influence coefficients
    Mx = 0.5 * Cx * log_term + ((Dx - A*Cx) / E_safe) * atan_term
    My = 0.5 * Cy * log_term + ((Dy - A*Cy) / E_safe) * atan_term
    
    # Zero out singular points
    if E < 1e-12:
        Mx = 0.0
        My = 0.0
    
    return Mx, My
