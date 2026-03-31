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

# Small number for numerical stability
EPS = 1e-12


def compute_quad_source_potential(
    point: NDArray[np.float64],
    vertices: NDArray[np.float64],
    sigma: float = 1.0,
) -> float:
    """
    Compute velocity potential at a point due to a quad source panel.
    
    Uses K&P Eq. 10.89 for the near-field and Eq. 10.100 for far-field.
    
    Args:
        point: (3,) field point in panel-local coordinates
        vertices: (4, 3) panel corner vertices in CCW order (panel-local)
        sigma: Source strength (default 1.0 for unit influence)
    
    Returns:
        Velocity potential Φ at the point
    """
    x, y, z = point[0], point[1], point[2]
    
    # Extract vertex coordinates (in panel-local frame, z_k = 0)
    x1, y1 = vertices[0, 0], vertices[0, 1]
    x2, y2 = vertices[1, 0], vertices[1, 1]
    x3, y3 = vertices[2, 0], vertices[2, 1]
    x4, y4 = vertices[3, 0], vertices[3, 1]
    
    # Edge lengths (Eq. 10.90)
    d12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    d23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    d34 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
    d41 = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
    
    # Edge slopes (Eq. 10.91) - handle vertical edges
    m12 = _safe_slope(y2 - y1, x2 - x1)
    m23 = _safe_slope(y3 - y2, x3 - x2)
    m34 = _safe_slope(y4 - y3, x4 - x3)
    m41 = _safe_slope(y1 - y4, x1 - x4)
    
    # Distances from point to vertices (Eq. 10.92)
    r1 = np.sqrt((x - x1)**2 + (y - y1)**2 + z**2)
    r2 = np.sqrt((x - x2)**2 + (y - y2)**2 + z**2)
    r3 = np.sqrt((x - x3)**2 + (y - y3)**2 + z**2)
    r4 = np.sqrt((x - x4)**2 + (y - y4)**2 + z**2)
    
    # Auxiliary quantities (Eq. 10.93, 10.94)
    e1 = (x - x1)**2 + z**2
    e2 = (x - x2)**2 + z**2
    e3 = (x - x3)**2 + z**2
    e4 = (x - x4)**2 + z**2
    
    h1 = (x - x1) * (y - y1)
    h2 = (x - x2) * (y - y2)
    h3 = (x - x3) * (y - y3)
    h4 = (x - x4) * (y - y4)
    
    # Compute potential using Eq. 10.89
    # First part: logarithmic terms from each edge
    phi = 0.0
    
    # Edge 1-2
    phi += _log_term(x, y, x1, y1, x2, y2, d12, r1, r2)
    # Edge 2-3
    phi += _log_term(x, y, x2, y2, x3, y3, d23, r2, r3)
    # Edge 3-4
    phi += _log_term(x, y, x3, y3, x4, y4, d34, r3, r4)
    # Edge 4-1
    phi += _log_term(x, y, x4, y4, x1, y1, d41, r4, r1)
    
    # Second part: arctangent terms (solid angle contribution)
    if abs(z) > EPS:
        atan_sum = 0.0
        atan_sum += np.arctan2(m12 * e1 - h1, z * r1) - np.arctan2(m12 * e2 - h2, z * r2)
        atan_sum += np.arctan2(m23 * e2 - h2, z * r2) - np.arctan2(m23 * e3 - h3, z * r3)
        atan_sum += np.arctan2(m34 * e3 - h3, z * r3) - np.arctan2(m34 * e4 - h4, z * r4)
        atan_sum += np.arctan2(m41 * e4 - h4, z * r4) - np.arctan2(m41 * e1 - h1, z * r1)
        
        phi -= abs(z) * atan_sum
    
    return -sigma / (4 * np.pi) * phi


def compute_quad_source_velocity(
    point: NDArray[np.float64],
    vertices: NDArray[np.float64],
    sigma: float = 1.0,
) -> NDArray[np.float64]:
    """
    Compute velocity at a point due to a quad source panel.
    
    Uses K&P Eq. 10.95-10.97 for velocity components.
    
    Args:
        point: (3,) field point in panel-local coordinates
        vertices: (4, 3) panel corner vertices in CCW order (panel-local)
        sigma: Source strength (default 1.0 for unit influence)
    
    Returns:
        (3,) velocity vector [u, v, w] in panel-local coordinates
    """
    x, y, z = point[0], point[1], point[2]
    
    # Extract vertex coordinates
    x1, y1 = vertices[0, 0], vertices[0, 1]
    x2, y2 = vertices[1, 0], vertices[1, 1]
    x3, y3 = vertices[2, 0], vertices[2, 1]
    x4, y4 = vertices[3, 0], vertices[3, 1]
    
    # Edge lengths
    d12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) + EPS
    d23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2) + EPS
    d34 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2) + EPS
    d41 = np.sqrt((x1 - x4)**2 + (y1 - y4)**2) + EPS
    
    # Edge slopes
    m12 = _safe_slope(y2 - y1, x2 - x1)
    m23 = _safe_slope(y3 - y2, x3 - x2)
    m34 = _safe_slope(y4 - y3, x4 - x3)
    m41 = _safe_slope(y1 - y4, x1 - x4)
    
    # Distances to vertices
    r1 = np.sqrt((x - x1)**2 + (y - y1)**2 + z**2) + EPS
    r2 = np.sqrt((x - x2)**2 + (y - y2)**2 + z**2) + EPS
    r3 = np.sqrt((x - x3)**2 + (y - y3)**2 + z**2) + EPS
    r4 = np.sqrt((x - x4)**2 + (y - y4)**2 + z**2) + EPS
    
    # Auxiliary quantities
    e1, e2 = (x - x1)**2 + z**2, (x - x2)**2 + z**2
    e3, e4 = (x - x3)**2 + z**2, (x - x4)**2 + z**2
    h1, h2 = (x - x1) * (y - y1), (x - x2) * (y - y2)
    h3, h4 = (x - x3) * (y - y3), (x - x4) * (y - y4)
    
    # u component (Eq. 10.95)
    u = ((y2 - y1) / d12 * _log_arg(r1, r2, d12) +
         (y3 - y2) / d23 * _log_arg(r2, r3, d23) +
         (y4 - y3) / d34 * _log_arg(r3, r4, d34) +
         (y1 - y4) / d41 * _log_arg(r4, r1, d41))
    
    # v component (Eq. 10.96)
    v = ((x1 - x2) / d12 * _log_arg(r1, r2, d12) +
         (x2 - x3) / d23 * _log_arg(r2, r3, d23) +
         (x3 - x4) / d34 * _log_arg(r3, r4, d34) +
         (x4 - x1) / d41 * _log_arg(r4, r1, d41))
    
    # w component (Eq. 10.97) - normal velocity
    w = (_atan_term(m12, e1, h1, z, r1) - _atan_term(m12, e2, h2, z, r2) +
         _atan_term(m23, e2, h2, z, r2) - _atan_term(m23, e3, h3, z, r3) +
         _atan_term(m34, e3, h3, z, r3) - _atan_term(m34, e4, h4, z, r4) +
         _atan_term(m41, e4, h4, z, r4) - _atan_term(m41, e1, h1, z, r1))
    
    coeff = sigma / (4 * np.pi)
    return np.array([coeff * u, coeff * v, coeff * w])


def compute_quad_source_velocity_vectorized(
    point: NDArray[np.float64],
    all_vertices: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute total velocity at a point from all quad source panels (vectorized).
    
    Args:
        point: (3,) field point in global coordinates
        all_vertices: (N, 4, 3) all panel vertices in global coordinates
        sigma: (N,) source strengths
    
    Returns:
        (3,) total velocity vector
    """
    n_panels = len(sigma)
    velocity = np.zeros(3, dtype=np.float64)
    
    for j in range(n_panels):
        v_local = compute_quad_source_velocity(point, all_vertices[j], sigma[j])
        velocity += v_local
    
    return velocity


def compute_source_influence_matrix(
    centers: NDArray[np.float64],
    normals: NDArray[np.float64],
    vertices: NDArray[np.float64],
    panels: NDArray[np.int32],
) -> NDArray[np.float64]:
    """
    Compute the source influence matrix for normal velocity BC.
    
    Element [i, j] = normal velocity at panel i center due to unit source on panel j.
    
    The boundary condition V·n = 0 gives: Σ_j A_ij σ_j = -V_∞·n_i
    
    For self-influence (i==j), the normal velocity jump is ±σ/2 (K&P Eq. 10.98).
    
    Args:
        centers: (N, 3) panel center points
        normals: (N, 3) panel outward normals
        vertices: (M, 3) all mesh nodes
        panels: (N, 4) panel connectivity
    
    Returns:
        (N, N) influence matrix A
    """
    n_panels = centers.shape[0]
    A = np.zeros((n_panels, n_panels), dtype=np.float64)
    
    for i in range(n_panels):
        point = centers[i]
        normal_i = normals[i]
        
        for j in range(n_panels):
            # Get panel j vertices
            panel_verts = vertices[panels[j]]  # (4, 3)
            
            if i == j:
                # Self-influence: normal velocity jump = σ/2 (K&P Eq. 10.98)
                # For a source panel, the induced normal velocity on itself is -σ/2
                # (factor of -1/2 because we're computing influence coefficient,
                # and the jump is σ/2 on each side, total discontinuity σ)
                A[i, j] = -0.5
            else:
                # Transform point to panel j local coordinates
                point_local, panel_verts_local = _to_panel_local(
                    point, panel_verts
                )
                
                # Compute velocity in panel-local frame
                vel_local = compute_quad_source_velocity(
                    point_local, panel_verts_local, sigma=1.0
                )
                
                # Transform velocity back to global frame
                vel_global = _velocity_to_global(vel_local, panel_verts)
                
                # Normal component
                A[i, j] = np.dot(vel_global, normal_i)
    
    return A


def compute_source_velocity_influence(
    point: NDArray[np.float64],
    centers: NDArray[np.float64],
    vertices: NDArray[np.float64],
    panels: NDArray[np.int32],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute induced velocity at a field point from all source panels.
    
    Args:
        point: (3,) field point in global coordinates
        centers: (N, 3) panel centers (not used directly, for consistency)
        vertices: (M, 3) all mesh nodes
        panels: (N, 4) panel connectivity
        sigma: (N,) source strengths
    
    Returns:
        (3,) induced velocity vector in global frame
    """
    n_panels = len(sigma)
    velocity = np.zeros(3, dtype=np.float64)
    
    for j in range(n_panels):
        panel_verts = vertices[panels[j]]  # (4, 3)
        
        # Transform to panel-local
        point_local, panel_verts_local = _to_panel_local(point, panel_verts)
        
        # Compute local velocity
        vel_local = compute_quad_source_velocity(
            point_local, panel_verts_local, sigma[j]
        )
        
        # Transform back to global
        vel_global = _velocity_to_global(vel_local, panel_verts)
        velocity += vel_global
    
    return velocity


# --- Helper functions ---

def _safe_slope(dy: float, dx: float) -> float:
    """Compute slope handling vertical edges."""
    if abs(dx) < EPS:
        return 1e10 * np.sign(dy) if abs(dy) > EPS else 0.0
    return dy / dx


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


def _log_arg(r1: float, r2: float, d: float) -> float:
    """Compute log argument for velocity formula."""
    denom_minus = r1 + r2 - d
    denom_plus = r1 + r2 + d
    
    if abs(denom_plus) < EPS or denom_plus <= 0:
        return 0.0
    if abs(denom_minus) < EPS or denom_minus <= 0:
        return 0.0
    
    return np.log(denom_minus / denom_plus)


def _atan_term(m: float, e: float, h: float, z: float, r: float) -> float:
    """Compute arctangent term for w velocity."""
    if abs(z) < EPS or abs(r) < EPS:
        return 0.0
    return np.arctan2(m * e - h, z * r)


def _to_panel_local(
    point: NDArray[np.float64],
    panel_verts: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Transform point and panel vertices to panel-local coordinates.
    
    Local frame: origin at panel center, z-axis = panel normal,
    x-axis along first edge direction.
    
    Args:
        point: (3,) point in global frame
        panel_verts: (4, 3) panel vertices in global frame
    
    Returns:
        (point_local, verts_local) both in panel-local frame
    """
    # Panel center
    center = np.mean(panel_verts, axis=0)
    
    # Panel normal (from diagonal cross product, d2 × d1 for outward)
    d1 = panel_verts[2] - panel_verts[0]
    d2 = panel_verts[3] - panel_verts[1]
    normal = np.cross(d2, d1)  # Note: d2 × d1 for outward normal
    normal = normal / (np.linalg.norm(normal) + EPS)
    
    # Local x-axis: first edge direction
    edge1 = panel_verts[1] - panel_verts[0]
    edge1_mag = np.linalg.norm(edge1)
    
    if edge1_mag > EPS:
        local_x = edge1 / edge1_mag
    else:
        # Degenerate first edge (e.g. at sphere poles), use second edge
        edge2 = panel_verts[2] - panel_verts[1]
        local_x = edge2 / (np.linalg.norm(edge2) + EPS)
    
    # Local y-axis: perpendicular to normal and x
    local_y = np.cross(normal, local_x)
    local_y = local_y / (np.linalg.norm(local_y) + EPS)
    
    # Rotation matrix (columns are local axes in global frame)
    R = np.column_stack([local_x, local_y, normal])
    
    # Transform point
    point_rel = point - center
    point_local = R.T @ point_rel
    
    # Transform vertices
    verts_rel = panel_verts - center
    verts_local = (R.T @ verts_rel.T).T
    
    return point_local, verts_local


def _velocity_to_global(
    vel_local: NDArray[np.float64],
    panel_verts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Transform velocity from panel-local to global coordinates.
    
    Args:
        vel_local: (3,) velocity in panel-local frame
        panel_verts: (4, 3) panel vertices in global frame (for computing axes)
    
    Returns:
        (3,) velocity in global frame
    """
    # Reconstruct local frame axes
    d1 = panel_verts[2] - panel_verts[0]
    d2 = panel_verts[3] - panel_verts[1]
    normal = np.cross(d2, d1)  # Note: d2 × d1 for outward normal
    normal = normal / (np.linalg.norm(normal) + EPS)
    
    edge1 = panel_verts[1] - panel_verts[0]
    edge1_mag = np.linalg.norm(edge1)
    
    if edge1_mag > EPS:
        local_x = edge1 / edge1_mag
    else:
        # Degenerate first edge, use second
        edge2 = panel_verts[2] - panel_verts[1]
        local_x = edge2 / (np.linalg.norm(edge2) + EPS)
    
    local_y = np.cross(normal, local_x)
    local_y = local_y / (np.linalg.norm(local_y) + EPS)
    
    # Rotation matrix
    R = np.column_stack([local_x, local_y, normal])
    
    # Transform velocity
    return R @ vel_local
