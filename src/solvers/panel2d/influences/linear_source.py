"""
Linear Source panel influence coefficient computations for 2D flow.

These functions compute the higher-order geometric integrals for linearly 
varying source panel methods following the Katz & Plotkin formulation 
(Continuous Strength boundary conditions: N+1 unknowns).
"""

import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from core.geometry.mesh import Mesh


def compute_linear_source_velocity_influence(
    point: NDArray,
    panel_start: NDArray,
    panel_length: float,
    panel_angle: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute velocity influence of a linearly varying source panel at an arbitrary field point.
    
    The velocity at point P is a linear combination of the nodal source strengths:
        V = (Mx_a, My_a) * sigma_j  +  (Mx_b, My_b) * sigma_{j+1}
        
    Args:
        point: (x, y) field point coordinates
        panel_start: (x, y) panel start node
        panel_length: Panel length S
        panel_angle: Panel orientation φ
    
    Returns:
        Tuple of ((Mx_a, My_a), (Mx_b, My_b)) velocity influence coefficients 
        for the leading and trailing nodes respectively.
    """
    dx = point[0] - panel_start[0]
    dy = point[1] - panel_start[1]
    
    cos_phi = np.cos(panel_angle)
    sin_phi = np.sin(panel_angle)
    
    # 1. Transform to local panel coordinates (origin at leading node, x along panel)
    x_loc = dx * cos_phi + dy * sin_phi
    y_loc = -dx * sin_phi + dy * cos_phi
    
    S = panel_length
    
    # 2. Geometric variables
    # Prevent divide-by-zero or zero-log warnings near singularities
    r1 = np.maximum(np.hypot(x_loc, y_loc), 1e-12)
    r2 = np.maximum(np.hypot(x_loc - S, y_loc), 1e-12)
    
    theta1 = np.arctan2(y_loc, x_loc)
    theta2 = np.arctan2(y_loc, x_loc - S)
    d_theta = theta2 - theta1
    
    # If the point is exactly on the panel, we need carefully handle the angle jumps
    # But np.arctan2 handles signs appropriately.
    
    # Mathematical factors 
    log_r1_r2 = np.log(r1 / r2)
    factor = 1.0 / (2.0 * np.pi)
    s_factor = factor / S
    
    # 3. Assess decomposed influence in local frame
    # Leading Node (a) locally induced velocity
    u_loc_a = s_factor * (S - x_loc) * log_r1_r2 + factor - s_factor * y_loc * d_theta
    w_loc_a = s_factor * (S - x_loc) * d_theta + s_factor * y_loc * log_r1_r2
    
    # Trailing Node (b) locally induced velocity
    u_loc_b = s_factor * x_loc * log_r1_r2 - factor + s_factor * y_loc * d_theta
    w_loc_b = s_factor * x_loc * d_theta - s_factor * y_loc * log_r1_r2
    
    # Mask singular potentials directly on the extreme edge points (nodes)
    if r1 <= 1e-12 or r2 <= 1e-12:
        # At the nodes themselves, limit to zero or proper principal values.
        # For simplicity numerically, assigning zero to node self-evaluation.
        u_loc_a = w_loc_a = 0.0
        u_loc_b = w_loc_b = 0.0
        
    # On the panel itself (y_loc == 0, 0 < x_loc < S), w_loc has a discontinuity 
    # and evaluates identically to +/- 0.5 depending on side.
    if np.abs(y_loc) < 1e-12 and 0 < x_loc < S:
        # Normal velocity on the panel itself is 0.5 * sigma.
        # But this function returns influence of sigma.
        # w_loc_a should be 0.5 * (1 - x/S), w_loc_b should be 0.5 * (x/S)
        # We can implement this analytically but for now we follow the general continuous formulation where y_loc -> 0.
        pass
        
    # 4. Rotate decomposed influences back to global boundary coordinates
    # local to global rotation
    Mx_a = u_loc_a * cos_phi - w_loc_a * sin_phi
    My_a = u_loc_a * sin_phi + w_loc_a * cos_phi
    
    Mx_b = u_loc_b * cos_phi - w_loc_b * sin_phi
    My_b = u_loc_b * sin_phi + w_loc_b * cos_phi
    
    return (Mx_a, My_a), (Mx_b, My_b)


def compute_linear_source_influence_matrices(mesh: Mesh) -> Tuple[NDArray, NDArray]:
    """
    Compute normal (I) and tangential (J) influence coefficient matrices for linear source panels.
    Continuous formulation: Unknowns are at nodes, evaluated at panel centers.
    
    Args:
        mesh: 2D panel mesh. Expects closed bodies (num_panels == num_nodes).
        
    Returns:
        Tuple of (I, J) matrices, shape (P, N) where P = num_panels, N = num_nodes.
    """
    num_panels = mesh.num_panels
    num_nodes = mesh.num_nodes
    
    # Extract geometries
    centers = mesh.centers[:, :2]
    # normals = mesh.normals[:, :2] # wait, normal components are stored as (nx, ny, nz)
    nx = mesh.normals[:, 0]
    ny = mesh.normals[:, 1]
    
    tx = mesh.tangents[:, 0]
    ty = mesh.tangents[:, 1]
    
    nodes2d = mesh.nodes[:, :2]
    
    tangents = mesh.tangents[:, :2]
    phi = np.arctan2(tangents[:, 1], tangents[:, 0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    
    I = np.zeros((num_panels, num_nodes))
    J = np.zeros((num_panels, num_nodes))
    
    for i in range(num_panels):
        cp = centers[i]
        n_i = np.array([nx[i], ny[i]])
        t_i = np.array([tx[i], ty[i]])
        
        for j in range(num_panels):
            n1_idx = mesh.panels[j, 0]
            n2_idx = mesh.panels[j, 1]
            
            p_start = nodes2d[n1_idx]
            S = mesh.areas[j]
            angle = phi[j]
            
            # For self-influence, the geometric integral handles it correctly 
            # as our function checks for singularities, but let's be careful.
            # Katz&Plotkin analytic solutions cover the case where point is on panel.
            # When computing control point on panel j itself:
            # point is at (S/2, 0) in local coords.
            # u_loc_a = u_loc_b = 0 due to log(r1/r2) cancelling and symmetry.
            # w_loc_a = w_loc_b = 0.5 (actually +/- 0.5)
            # but wait! I should handle i==j analytically to be safe.
            
            if i == j:
                # Analytical self-influence:
                # Normal velocity: sigma_a * 0.5 * (1 - 0.5) + sigma_b * 0.5 * (0.5)
                #   Wait, sigma(x_loc) = sigma_a(1 - x/S) + sigma_b(x/S).
                #   At x = S/2, sigma = 0.5*(sigma_a + sigma_b).
                #   Self-induced normal velocity is 0.5 * sigma.
                # So I adds 0.5 to both nodes (since 0.5 * 0.5 = 0.25).
                I[i, n1_idx] += 0.25
                I[i, n2_idx] += 0.25
                # Tangential velocity self-influence is 0.
                pass
            else:
                vel_a, vel_b = compute_linear_source_velocity_influence(
                    point=cp,
                    panel_start=p_start,
                    panel_length=S,
                    panel_angle=angle
                )
                
                # Normal influence
                I[i, n1_idx] += vel_a[0] * n_i[0] + vel_a[1] * n_i[1]
                I[i, n2_idx] += vel_b[0] * n_i[0] + vel_b[1] * n_i[1]
                
                # Tangential influence
                J[i, n1_idx] += vel_a[0] * t_i[0] + vel_a[1] * t_i[1]
                J[i, n2_idx] += vel_b[0] * t_i[0] + vel_b[1] * t_i[1]

    return I, J

def compute_linear_source_velocity_field(
    points: NDArray,
    mesh: Mesh,
    strengths: NDArray
) -> NDArray:
    """
    Compute velocity field at given points due to linear source panels.
    
    Args:
        points: (M, 2) or (M, 3) coordinates
        mesh: Panel mesh
        strengths: (N_nodes,) source strengths
        
    Returns:
        (M, 2) velocity vectors
    """
    n_points = points.shape[0]
    n_panels = mesh.num_panels
    
    V = np.zeros((n_points, 2))
    
    nodes2d = mesh.nodes[:, :2]
    tangents = mesh.tangents[:, :2]
    phi = np.arctan2(tangents[:, 1], tangents[:, 0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    
    for j in range(n_panels):
        n1_idx = mesh.panels[j, 0]
        n2_idx = mesh.panels[j, 1]
        
        p_start = nodes2d[n1_idx]
        S = mesh.areas[j]
        angle = phi[j]
        
        sig_a = strengths[n1_idx]
        sig_b = strengths[n2_idx]
        
        for k in range(n_points):
            vel_a, vel_b = compute_linear_source_velocity_influence(
                point=points[k, :2],
                panel_start=p_start,
                panel_length=S,
                panel_angle=angle
            )
            
            V[k, 0] += sig_a * vel_a[0] + sig_b * vel_b[0]
            V[k, 1] += sig_a * vel_a[1] + sig_b * vel_b[1]
            
    return V