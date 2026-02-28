"""
Linear Vortex panel influence coefficient computations for 2D flow.

These functions compute the higher-order geometric integrals for linearly 
varying vortex panel methods. Adapted for pure bluff bodies using 
Zero Net Circulation closure (N+1 unknowns).
"""

import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from core.geometry.mesh import Mesh

def compute_linear_vortex_velocity_influence(
    point: NDArray,
    panel_start: NDArray,
    panel_length: float,
    panel_angle: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute velocity influence of a linearly varying vortex panel at an arbitrary field point.
    
    Returns:
        Tuple of ((Mx_a, My_a), (Mx_b, My_b)) velocity influence coefficients 
        for the leading and trailing nodes respectively.
    """
    dx = point[0] - panel_start[0]
    dy = point[1] - panel_start[1]
    
    cos_phi = np.cos(panel_angle)
    sin_phi = np.sin(panel_angle)
    
    # Transform to local panel coordinates
    x_loc = dx * cos_phi + dy * sin_phi
    y_loc = -dx * sin_phi + dy * cos_phi
    
    S = panel_length
    
    r1 = np.maximum(np.hypot(x_loc, y_loc), 1e-12)
    r2 = np.maximum(np.hypot(x_loc - S, y_loc), 1e-12)
    
    theta1 = np.arctan2(y_loc, x_loc)
    theta2 = np.arctan2(y_loc, x_loc - S)
    d_theta = theta2 - theta1
    
    log_r1_r2 = np.log(r1 / r2)
    factor = 1.0 / (2.0 * np.pi)
    s_factor = factor / S
    
    # Assess decomposed influence in local frame for SOURCE (Katz Eq 10.48/10.49 mapped)
    u_loc_source_a = s_factor * (S - x_loc) * log_r1_r2 + factor - s_factor * y_loc * d_theta
    w_loc_source_a = s_factor * (S - x_loc) * d_theta + s_factor * y_loc * log_r1_r2
    
    u_loc_source_b = s_factor * x_loc * log_r1_r2 - factor + s_factor * y_loc * d_theta
    w_loc_source_b = s_factor * x_loc * d_theta - s_factor * y_loc * log_r1_r2
    
    # Map source influence to vortex influence
    # u_vortex = w_source  ;  w_vortex = -u_source
    u_loc_a = w_loc_source_a
    w_loc_a = -u_loc_source_a
    
    u_loc_b = w_loc_source_b
    w_loc_b = -u_loc_source_b
    
    if r1 <= 1e-12 or r2 <= 1e-12:
        u_loc_a = w_loc_a = 0.0
        u_loc_b = w_loc_b = 0.0
        
    # Rotate back to global boundary coordinates
    Mx_a = u_loc_a * cos_phi - w_loc_a * sin_phi
    My_a = u_loc_a * sin_phi + w_loc_a * cos_phi
    
    Mx_b = u_loc_b * cos_phi - w_loc_b * sin_phi
    My_b = u_loc_b * sin_phi + w_loc_b * cos_phi
    
    return (Mx_a, My_a), (Mx_b, My_b)

def compute_linear_vortex_influence_matrices(mesh: Mesh) -> Tuple[NDArray, NDArray]:
    """
    Compute normal (I) and tangential (J) influence coefficient matrices for linear vortex panels.
    Continuous formulation: Unknowns are at nodes, evaluated at panel centers.
    """
    num_panels = mesh.num_panels
    num_nodes = mesh.num_nodes
    
    centers = mesh.centers[:, :2]
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
            
            if i == j:
                # Geometric exactness safely determined by pushing cp slightly OUTWARD 
                # (along mesh outward normal n_i) into the fluid domain.
                cp_eval = cp + 1e-10 * n_i
            else:
                cp_eval = cp
                
            vel_a, vel_b = compute_linear_vortex_velocity_influence(
                point=cp_eval,
                panel_start=p_start,
                panel_length=S,
                panel_angle=angle
            )
            
            # Normal influence (I matrix)
            I[i, n1_idx] += vel_a[0] * n_i[0] + vel_a[1] * n_i[1]
            I[i, n2_idx] += vel_b[0] * n_i[0] + vel_b[1] * n_i[1]
            
            # Tangential influence (J matrix)
            J[i, n1_idx] += vel_a[0] * t_i[0] + vel_a[1] * t_i[1]
            J[i, n2_idx] += vel_b[0] * t_i[0] + vel_b[1] * t_i[1]

    return I, J

def compute_linear_vortex_velocity_field(
    points: NDArray,
    mesh: Mesh,
    strengths: NDArray
) -> NDArray:
    """
    Compute velocity field at given points due to linear vortex panels.
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
            vel_a, vel_b = compute_linear_vortex_velocity_influence(
                point=points[k, :2],
                panel_start=p_start,
                panel_length=S,
                panel_angle=angle
            )
            
            V[k, 0] += sig_a * vel_a[0] + sig_b * vel_b[0]
            V[k, 1] += sig_a * vel_a[1] + sig_b * vel_b[1]
            
    return V
