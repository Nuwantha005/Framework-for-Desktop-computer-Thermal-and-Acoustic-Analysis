"""
3D quadrilateral constant-strength source panel influence functions.

Implements Katz & Plotkin formulation (Eq. 10.89-10.103) for constant-strength
source distributions on flat quadrilateral panels.

All computations are performed in panel-local coordinates where:
- The panel lies in the z=0 plane
- Vertices are (x1,y1,0), (x2,y2,0), (x3,y3,0), (x4,y4,0) in CCW order
- The influence point P is at (x, y, z)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from numba import njit, prange

# Small number for numerical stability
EPS = 1e-12

@njit(fastmath=False, cache=True)
def _safe_slope(dy: float, dx: float) -> float:
    """Compute slope handling vertical edges."""
    if abs(dx) < EPS:
        return 1e10 * np.sign(dy) if abs(dy) > EPS else 0.0
    return dy / dx

@njit(fastmath=False, cache=True)
def _log_term(
    x: float, y: float,
    x1: float, y1: float,
    x2: float, y2: float,
    d: float, r1: float, r2: float
) -> float:
    """Compute logarithmic contribution from one edge."""
    # Numerator: (x-x1)(y2-y1) - (y-y1)(x2-x1)
    num = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    
    # Argument of log: (r1 + r2 + d) / (r1 + r2 - d)
    denom_plus = r1 + r2 + d
    denom_minus = r1 + r2 - d
    
    if abs(denom_minus) < EPS or denom_minus <= 0:
        return 0.0
    
    if d < EPS:
        return 0.0
    
    return (num / d) * np.log(denom_plus / denom_minus)

@njit(fastmath=False, cache=True)
def _log_arg(r1: float, r2: float, d: float) -> float:
    """Compute log argument for velocity formula."""
    denom_minus = r1 + r2 - d
    denom_plus = r1 + r2 + d
    
    if abs(denom_plus) < EPS or denom_plus <= 0:
        return 0.0
    if abs(denom_minus) < EPS or denom_minus <= 0:
        return 0.0
    
    return np.log(denom_minus / denom_plus)

@njit(fastmath=False, cache=True)
def _atan_term(m: float, e: float, h: float, z: float, r: float) -> float:
    """Compute arctangent term for w velocity."""
    if abs(z) < EPS or abs(r) < EPS:
        return 0.0
    return np.arctan2(m * e - h, z * r)

@njit(fastmath=False, cache=True)
def _to_panel_local(
    point: NDArray[np.float64],
    panel_verts: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Transform point and panel vertices to panel-local coordinates.
    """
    # Panel center
    center = np.zeros(3)
    for i in range(4):
        center += panel_verts[i]
    center /= 4.0
    
    # Panel normal
    d1 = panel_verts[2] - panel_verts[0]
    d2 = panel_verts[3] - panel_verts[1]
    
    normal = np.empty(3)
    normal[0] = d2[1]*d1[2] - d2[2]*d1[1]
    normal[1] = d2[2]*d1[0] - d2[0]*d1[2]
    normal[2] = d2[0]*d1[1] - d2[1]*d1[0]
    
    n_norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2) + EPS
    normal /= n_norm
    
    # Local x-axis
    edge1 = panel_verts[1] - panel_verts[0]
    edge1_mag = np.sqrt(edge1[0]**2 + edge1[1]**2 + edge1[2]**2)
    
    local_x = np.empty(3)
    if edge1_mag > EPS:
        local_x = edge1 / edge1_mag
    else:
        edge2 = panel_verts[2] - panel_verts[1]
        e2_norm = np.sqrt(edge2[0]**2 + edge2[1]**2 + edge2[2]**2) + EPS
        local_x = edge2 / e2_norm
    
    # Local y-axis
    local_y = np.empty(3)
    local_y[0] = normal[1]*local_x[2] - normal[2]*local_x[1]
    local_y[1] = normal[2]*local_x[0] - normal[0]*local_x[2]
    local_y[2] = normal[0]*local_x[1] - normal[1]*local_x[0]
    
    y_norm = np.sqrt(local_y[0]**2 + local_y[1]**2 + local_y[2]**2) + EPS
    local_y /= y_norm
    
    # Rotation matrix (R.T)
    RT = np.empty((3, 3))
    RT[0, 0] = local_x[0]; RT[0, 1] = local_x[1]; RT[0, 2] = local_x[2]
    RT[1, 0] = local_y[0]; RT[1, 1] = local_y[1]; RT[1, 2] = local_y[2]
    RT[2, 0] = normal[0]; RT[2, 1] = normal[1]; RT[2, 2] = normal[2]
    
    # Transform point
    point_rel = point - center
    point_local = np.empty(3)
    point_local[0] = RT[0,0]*point_rel[0] + RT[0,1]*point_rel[1] + RT[0,2]*point_rel[2]
    point_local[1] = RT[1,0]*point_rel[0] + RT[1,1]*point_rel[1] + RT[1,2]*point_rel[2]
    point_local[2] = RT[2,0]*point_rel[0] + RT[2,1]*point_rel[1] + RT[2,2]*point_rel[2]
    
    # Transform vertices
    verts_local = np.empty((4, 3))
    for i in range(4):
        v_rel = panel_verts[i] - center
        verts_local[i, 0] = RT[0,0]*v_rel[0] + RT[0,1]*v_rel[1] + RT[0,2]*v_rel[2]
        verts_local[i, 1] = RT[1,0]*v_rel[0] + RT[1,1]*v_rel[1] + RT[1,2]*v_rel[2]
        verts_local[i, 2] = RT[2,0]*v_rel[0] + RT[2,1]*v_rel[1] + RT[2,2]*v_rel[2]
        
    return point_local, verts_local

@njit(fastmath=False, cache=True)
def _velocity_to_global(
    vel_local: NDArray[np.float64],
    panel_verts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform velocity from panel-local to global coordinates."""
    d1 = panel_verts[2] - panel_verts[0]
    d2 = panel_verts[3] - panel_verts[1]
    
    normal = np.empty(3)
    normal[0] = d2[1]*d1[2] - d2[2]*d1[1]
    normal[1] = d2[2]*d1[0] - d2[0]*d1[2]
    normal[2] = d2[0]*d1[1] - d2[1]*d1[0]
    n_norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2) + EPS
    normal /= n_norm
    
    edge1 = panel_verts[1] - panel_verts[0]
    edge1_mag = np.sqrt(edge1[0]**2 + edge1[1]**2 + edge1[2]**2)
    
    local_x = np.empty(3)
    if edge1_mag > EPS:
        local_x = edge1 / edge1_mag
    else:
        edge2 = panel_verts[2] - panel_verts[1]
        e2_norm = np.sqrt(edge2[0]**2 + edge2[1]**2 + edge2[2]**2) + EPS
        local_x = edge2 / e2_norm
    
    local_y = np.empty(3)
    local_y[0] = normal[1]*local_x[2] - normal[2]*local_x[1]
    local_y[1] = normal[2]*local_x[0] - normal[0]*local_x[2]
    local_y[2] = normal[0]*local_x[1] - normal[1]*local_x[0]
    y_norm = np.sqrt(local_y[0]**2 + local_y[1]**2 + local_y[2]**2) + EPS
    local_y /= y_norm
    
    vel_global = np.empty(3)
    vel_global[0] = local_x[0]*vel_local[0] + local_y[0]*vel_local[1] + normal[0]*vel_local[2]
    vel_global[1] = local_x[1]*vel_local[0] + local_y[1]*vel_local[1] + normal[1]*vel_local[2]
    vel_global[2] = local_x[2]*vel_local[0] + local_y[2]*vel_local[1] + normal[2]*vel_local[2]
    
    return vel_global

@njit(fastmath=False, cache=True)
def compute_quad_source_potential(
    point: NDArray[np.float64],
    vertices: NDArray[np.float64],
    sigma: float = 1.0,
) -> float:
    """Compute velocity potential at a point due to a quad source panel."""
    x, y, z = point[0], point[1], point[2]
    
    x1, y1 = vertices[0, 0], vertices[0, 1]
    x2, y2 = vertices[1, 0], vertices[1, 1]
    x3, y3 = vertices[2, 0], vertices[2, 1]
    x4, y4 = vertices[3, 0], vertices[3, 1]
    
    d12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    d23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    d34 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
    d41 = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
    
    m12 = _safe_slope(y2 - y1, x2 - x1)
    m23 = _safe_slope(y3 - y2, x3 - x2)
    m34 = _safe_slope(y4 - y3, x4 - x3)
    m41 = _safe_slope(y1 - y4, x1 - x4)
    
    r1 = np.sqrt((x - x1)**2 + (y - y1)**2 + z**2)
    r2 = np.sqrt((x - x2)**2 + (y - y2)**2 + z**2)
    r3 = np.sqrt((x - x3)**2 + (y - y3)**2 + z**2)
    r4 = np.sqrt((x - x4)**2 + (y - y4)**2 + z**2)
    
    e1 = (x - x1)**2 + z**2
    e2 = (x - x2)**2 + z**2
    e3 = (x - x3)**2 + z**2
    e4 = (x - x4)**2 + z**2
    
    h1 = (x - x1) * (y - y1)
    h2 = (x - x2) * (y - y2)
    h3 = (x - x3) * (y - y3)
    h4 = (x - x4) * (y - y4)
    
    phi = 0.0
    phi += _log_term(x, y, x1, y1, x2, y2, d12, r1, r2)
    phi += _log_term(x, y, x2, y2, x3, y3, d23, r2, r3)
    phi += _log_term(x, y, x3, y3, x4, y4, d34, r3, r4)
    phi += _log_term(x, y, x4, y4, x1, y1, d41, r4, r1)
    
    if abs(z) > EPS:
        atan_sum = 0.0
        atan_sum += np.arctan2(m12 * e1 - h1, z * r1) - np.arctan2(m12 * e2 - h2, z * r2)
        atan_sum += np.arctan2(m23 * e2 - h2, z * r2) - np.arctan2(m23 * e3 - h3, z * r3)
        atan_sum += np.arctan2(m34 * e3 - h3, z * r3) - np.arctan2(m34 * e4 - h4, z * r4)
        atan_sum += np.arctan2(m41 * e4 - h4, z * r4) - np.arctan2(m41 * e1 - h1, z * r1)
        
        phi -= abs(z) * atan_sum
    
    return -sigma / (4 * np.pi) * phi

@njit(fastmath=False, cache=True)
def compute_quad_source_velocity(
    point: NDArray[np.float64],
    vertices: NDArray[np.float64],
    sigma: float = 1.0,
) -> NDArray[np.float64]:
    """Compute velocity at a point due to a quad source panel."""
    x, y, z = point[0], point[1], point[2]
    
    x1, y1 = vertices[0, 0], vertices[0, 1]
    x2, y2 = vertices[1, 0], vertices[1, 1]
    x3, y3 = vertices[2, 0], vertices[2, 1]
    x4, y4 = vertices[3, 0], vertices[3, 1]
    
    d12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) + EPS
    d23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2) + EPS
    d34 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2) + EPS
    d41 = np.sqrt((x1 - x4)**2 + (y1 - y4)**2) + EPS
    
    m12 = _safe_slope(y2 - y1, x2 - x1)
    m23 = _safe_slope(y3 - y2, x3 - x2)
    m34 = _safe_slope(y4 - y3, x4 - x3)
    m41 = _safe_slope(y1 - y4, x1 - x4)
    
    r1 = np.sqrt((x - x1)**2 + (y - y1)**2 + z**2) + EPS
    r2 = np.sqrt((x - x2)**2 + (y - y2)**2 + z**2) + EPS
    r3 = np.sqrt((x - x3)**2 + (y - y3)**2 + z**2) + EPS
    r4 = np.sqrt((x - x4)**2 + (y - y4)**2 + z**2) + EPS
    
    e1, e2 = (x - x1)**2 + z**2, (x - x2)**2 + z**2
    e3, e4 = (x - x3)**2 + z**2, (x - x4)**2 + z**2
    h1, h2 = (x - x1) * (y - y1), (x - x2) * (y - y2)
    h3, h4 = (x - x3) * (y - y3), (x - x4) * (y - y4)
    
    u = ((y2 - y1) / d12 * _log_arg(r1, r2, d12) +
         (y3 - y2) / d23 * _log_arg(r2, r3, d23) +
         (y4 - y3) / d34 * _log_arg(r3, r4, d34) +
         (y1 - y4) / d41 * _log_arg(r4, r1, d41))
    
    v = ((x1 - x2) / d12 * _log_arg(r1, r2, d12) +
         (x2 - x3) / d23 * _log_arg(r2, r3, d23) +
         (x3 - x4) / d34 * _log_arg(r3, r4, d34) +
         (x4 - x1) / d41 * _log_arg(r4, r1, d41))
    
    w = (_atan_term(m12, e1, h1, z, r1) - _atan_term(m12, e2, h2, z, r2) +
         _atan_term(m23, e2, h2, z, r2) - _atan_term(m23, e3, h3, z, r3) +
         _atan_term(m34, e3, h3, z, r3) - _atan_term(m34, e4, h4, z, r4) +
         _atan_term(m41, e4, h4, z, r4) - _atan_term(m41, e1, h1, z, r1))
    
    coeff = sigma / (4 * np.pi)
    res = np.empty(3)
    res[0] = coeff * u
    res[1] = coeff * v
    res[2] = coeff * w
    return res

@njit(fastmath=False, cache=True)
def compute_quad_source_velocity_vectorized(
    point: NDArray[np.float64],
    all_vertices: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_panels = len(sigma)
    velocity = np.zeros(3, dtype=np.float64)
    
    for j in range(n_panels):
        v_local = compute_quad_source_velocity(point, all_vertices[j], sigma[j])
        velocity += v_local
    
    return velocity

@njit(parallel=True, fastmath=False, cache=True)
def compute_source_influence_matrix(
    centers: NDArray[np.float64],
    normals: NDArray[np.float64],
    vertices: NDArray[np.float64],
    panels: NDArray[np.int32],
) -> NDArray[np.float64]:
    n_panels = centers.shape[0]
    A = np.zeros((n_panels, n_panels), dtype=np.float64)
    
    for i in prange(n_panels):
        point = centers[i]
        normal_i = normals[i]
        
        for j in range(n_panels):
            panel_verts = np.empty((4, 3))
            panel_verts[0] = vertices[panels[j, 0]]
            panel_verts[1] = vertices[panels[j, 1]]
            panel_verts[2] = vertices[panels[j, 2]]
            panel_verts[3] = vertices[panels[j, 3]]
            
            if i == j:
                A[i, j] = -0.5
            else:
                point_local, panel_verts_local = _to_panel_local(point, panel_verts)
                vel_local = compute_quad_source_velocity(point_local, panel_verts_local, 1.0)
                vel_global = _velocity_to_global(vel_local, panel_verts)
                A[i, j] = vel_global[0]*normal_i[0] + vel_global[1]*normal_i[1] + vel_global[2]*normal_i[2]
    
    return A

@njit(parallel=True, fastmath=False, cache=True)
def compute_all_velocities_influence(
    points: NDArray[np.float64],
    vertices: NDArray[np.float64],
    panels: NDArray[np.int32],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_points = points.shape[0]
    n_panels = len(sigma)
    velocities = np.zeros((n_points, 3), dtype=np.float64)
    
    for i in prange(n_points):
        point = points[i]
        for j in range(n_panels):
            panel_verts = np.empty((4, 3))
            panel_verts[0] = vertices[panels[j, 0]]
            panel_verts[1] = vertices[panels[j, 1]]
            panel_verts[2] = vertices[panels[j, 2]]
            panel_verts[3] = vertices[panels[j, 3]]
            
            point_local, panel_verts_local = _to_panel_local(point, panel_verts)
            vel_local = compute_quad_source_velocity(point_local, panel_verts_local, sigma[j])
            vel_global = _velocity_to_global(vel_local, panel_verts)
            velocities[i, 0] += vel_global[0]
            velocities[i, 1] += vel_global[1]
            velocities[i, 2] += vel_global[2]
            
    return velocities

