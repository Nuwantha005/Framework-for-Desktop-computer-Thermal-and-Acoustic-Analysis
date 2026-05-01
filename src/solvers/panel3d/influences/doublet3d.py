"""
3D quadrilateral constant-strength doublet panel influence functions.

A constant strength doublet panel is mathematically equivalent to a vortex ring
around its perimeter. This uses the Biot-Savart law for straight vortex segments.
"""

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

EPS = 1e-12

@njit(fastmath=False, cache=True)
def vortex_segment_velocity(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    point: NDArray[np.float64],
    gamma: float
) -> NDArray[np.float64]:
    """
    Compute velocity induced by a straight vortex segment.
    
    Args:
        p1: Start point of segment
        p2: End point of segment
        point: Evaluation point
        gamma: Circulation strength (equivalent to doublet strength mu)
        
    Returns:
        Induced velocity vector (3,)
    """
    r1 = point - p1
    r2 = point - p2
    
    # Cross product r1 x r2
    cross_x = r1[1]*r2[2] - r1[2]*r2[1]
    cross_y = r1[2]*r2[0] - r1[0]*r2[2]
    cross_z = r1[0]*r2[1] - r1[1]*r2[0]
    
    cross_mag_sq = cross_x**2 + cross_y**2 + cross_z**2
    
    # If point is on the vortex segment line, velocity is zero
    if cross_mag_sq < EPS:
        return np.zeros(3, dtype=np.float64)
        
    r1_mag = np.sqrt(r1[0]**2 + r1[1]**2 + r1[2]**2)
    r2_mag = np.sqrt(r2[0]**2 + r2[1]**2 + r2[2]**2)
    
    r0 = p2 - p1
    term1 = (r0[0]*r1[0] + r0[1]*r1[1] + r0[2]*r1[2]) / r1_mag
    term2 = (r0[0]*r2[0] + r0[1]*r2[1] + r0[2]*r2[2]) / r2_mag
    
    coeff = (gamma / (4.0 * np.pi)) * (term1 - term2) / cross_mag_sq
    
    res = np.empty(3, dtype=np.float64)
    res[0] = coeff * cross_x
    res[1] = coeff * cross_y
    res[2] = coeff * cross_z
    return res

@njit(fastmath=False, cache=True)
def compute_quad_doublet_velocity(
    point: NDArray[np.float64],
    vertices: NDArray[np.float64],
    mu: float = 1.0,
) -> NDArray[np.float64]:
    """
    Compute velocity at a point due to a quad doublet panel.
    Equivalent to a vortex ring around the panel perimeter.
    """
    vel = np.zeros(3, dtype=np.float64)
    
    # 4 edges of the quad.
    # Note: Mesh3D geometry generators produce CW-ordered nodes for outward normals.
    # A CW vortex ring induces velocity OPPOSITE to the outward normal.
    # To make a positive doublet (dipole along normal) induce velocity ALONG the normal,
    # we must subtract the vortex segment velocity (reversing circulation sign).
    vel -= vortex_segment_velocity(vertices[0], vertices[1], point, mu)
    vel -= vortex_segment_velocity(vertices[1], vertices[2], point, mu)
    vel -= vortex_segment_velocity(vertices[2], vertices[3], point, mu)
    vel -= vortex_segment_velocity(vertices[3], vertices[0], point, mu)
    
    return vel

@njit(parallel=True, fastmath=False, cache=True)
def compute_all_doublet_velocities(
    points: NDArray[np.float64],
    vertices: NDArray[np.float64],
    panels: NDArray[np.int32],
    mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute velocity at multiple points due to multiple doublet panels.
    """
    n_points = points.shape[0]
    n_panels = len(mu)
    
    velocities = np.zeros((n_points, 3), dtype=np.float64)
    
    for i in prange(n_points):
        point = points[i]
        vx, vy, vz = 0.0, 0.0, 0.0
        
        for j in range(n_panels):
            p0 = vertices[panels[j, 0]]
            p1 = vertices[panels[j, 1]]
            p2 = vertices[panels[j, 2]]
            p3 = vertices[panels[j, 3]]
            
            mu_j = mu[j]
            
            v1 = vortex_segment_velocity(p0, p1, point, mu_j)
            v2 = vortex_segment_velocity(p1, p2, point, mu_j)
            v3 = vortex_segment_velocity(p2, p3, point, mu_j)
            v4 = vortex_segment_velocity(p3, p0, point, mu_j)
            
            # Subtract because CW panels induce flow opposite to normal
            vx -= v1[0] + v2[0] + v3[0] + v4[0]
            vy -= v1[1] + v2[1] + v3[1] + v4[1]
            vz -= v1[2] + v2[2] + v3[2] + v4[2]
            
        velocities[i, 0] = vx
        velocities[i, 1] = vy
        velocities[i, 2] = vz
        
    return velocities
