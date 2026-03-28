"""
Extract BDIM input data from viscous boundary layer field reconstruction.

This module provides functions to convert the (s, y_normal) grid from the
BL field reconstruction to physical Cartesian (x, y) coordinates required
by the BDIM solver.

The coordinate transformation is straightforward:
    x_domain = x_surface + y_normal * n_x
    y_domain = y_surface + y_normal * n_y

where (n_x, n_y) is the outward unit normal at each surface station.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .solver import BDIMInput

if TYPE_CHECKING:
    from solvers.boundary_layer.field import BLFieldData
    from solvers.boundary_layer.runner import BoundaryLayerPathResult


def compute_path_normals(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute outward unit normals and flow tangents for an open path (not closed body).
    
    Uses central differences for interior points and one-sided differences
    at endpoints. Assumes the path flows from stagnation outward, so the
    "outward" normal points away from the body interior. Tangents always
    point in the flow direction.
    
    Args:
        x: Path x-coordinates, shape (K,)
        y: Path y-coordinates, shape (K,)
    
    Returns:
        normals: Unit normal vectors, shape (K, 2)
        tangents: Unit tangent vectors pointing in flow direction, shape (K, 2)
    """
    K = len(x)
    normals = np.zeros((K, 2), dtype=np.float64)
    tangents = np.zeros((K, 2), dtype=np.float64)
    
    for i in range(K):
        # Central difference for interior, one-sided at ends
        if i == 0:
            tx = x[1] - x[0]
            ty = y[1] - y[0]
        elif i == K - 1:
            tx = x[-1] - x[-2]
            ty = y[-1] - y[-2]
        else:
            tx = x[i + 1] - x[i - 1]
            ty = y[i + 1] - y[i - 1]
        
        # Normalize tangent
        length = np.sqrt(tx**2 + ty**2)
        if length > 1e-12:
            tx /= length
            ty /= length
            
        tangents[i, 0] = tx
        tangents[i, 1] = ty
        
        # Rotate 90° to get normal (perpendicular to tangent)
        # For a path going left-to-right, (ty, -tx) points "up"
        normals[i, 0] = ty
        normals[i, 1] = -tx
    
    # Check normal direction using path curvature or assume upper/lower convention
    # For upper surface (y > 0 typically), normal should point up (positive y component)
    # For lower surface (y < 0 typically), normal should point down (negative y component)
    # We use a simple heuristic: the normal at the midpoint should point away from origin
    mid = K // 2
    mid_pt = np.array([x[mid], y[mid]])
    if np.dot(normals[mid], mid_pt) < 0:
        # Normals point toward origin, flip them
        normals = -normals
    
    return normals, tangents


def transform_sy_to_xy(
    s: NDArray[np.float64],
    y_normal: NDArray[np.float64],
    path_x: NDArray[np.float64],
    path_y: NDArray[np.float64],
    normals: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Transform (s, y_normal) grid to physical (x, y) Cartesian coordinates.
    
    For each station i and wall-normal distance j:
        x[i,j] = path_x[i] + y_normal[i,j] * normals[i,0]
        y[i,j] = path_y[i] + y_normal[i,j] * normals[i,1]
    
    Args:
        s: Arc-length stations, shape (M,) - not used but included for API clarity
        y_normal: Wall-normal distances, shape (M, Ny)
        path_x: Surface x-coordinates, shape (M,)
        path_y: Surface y-coordinates, shape (M,)
        normals: Outward unit normals, shape (M, 2)
    
    Returns:
        (x_domain, y_domain): Physical coordinates, each shape (M, Ny)
    """
    M, Ny = y_normal.shape
    x_domain = np.zeros((M, Ny), dtype=np.float64)
    y_domain = np.zeros((M, Ny), dtype=np.float64)
    
    for i in range(M):
        x_domain[i, :] = path_x[i] + y_normal[i, :] * normals[i, 0]
        y_domain[i, :] = path_y[i] + y_normal[i, :] * normals[i, 1]
    
    return x_domain, y_domain


def compute_velocity_gradients(
    u: NDArray[np.float64],
    y_normal: NDArray[np.float64],
    ds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute velocity gradients from reconstructed BL velocity field.
    
    In boundary layer coordinates:
        du/ds ≈ central difference in s direction
        du/dy ≈ central difference in y direction
        
    We assume v ≈ 0 in the thin boundary layer (standard BL approximation),
    so dv/ds ≈ 0 and dv/dy ≈ 0.
    
    The gradients are then transformed to Cartesian coordinates using the
    local tangent and normal vectors (simplified for thin BL).
    
    Args:
        u: Tangential velocity field, shape (M, Ny)
        y_normal: Wall-normal distances, shape (M, Ny)
        ds: Arc-length spacing, shape (M-1,) or scalar
    
    Returns:
        grad_u: Velocity gradient tensor, shape (M, Ny, 2, 2)
            Format: [[du/dx, du/dy], [dv/dx, dv/dy]]
            For thin BL: dv/* ≈ 0
    """
    M, Ny = u.shape
    grad_u = np.zeros((M, Ny, 2, 2), dtype=np.float64)
    
    # Compute du/ds using central differences
    du_ds = np.zeros_like(u)
    for i in range(M):
        if i == 0:
            # Forward difference
            if isinstance(ds, np.ndarray):
                h = ds[0]
            else:
                h = ds
            du_ds[i, :] = (u[1, :] - u[0, :]) / h
        elif i == M - 1:
            # Backward difference
            if isinstance(ds, np.ndarray):
                h = ds[-1]
            else:
                h = ds
            du_ds[i, :] = (u[-1, :] - u[-2, :]) / h
        else:
            # Central difference
            if isinstance(ds, np.ndarray):
                h = ds[i - 1] + ds[i]
            else:
                h = 2 * ds
            du_ds[i, :] = (u[i + 1, :] - u[i - 1, :]) / h
    
    # Compute du/dy using central differences in y direction
    du_dy = np.zeros_like(u)
    for i in range(M):
        y = y_normal[i, :]
        for j in range(Ny):
            if j == 0:
                # Forward difference (wall)
                dy = y[1] - y[0]
                if dy > 1e-12:
                    du_dy[i, j] = (u[i, 1] - u[i, 0]) / dy
            elif j == Ny - 1:
                # Backward difference (edge)
                dy = y[-1] - y[-2]
                if dy > 1e-12:
                    du_dy[i, j] = (u[i, -1] - u[i, -2]) / dy
            else:
                # Central difference
                dy = y[j + 1] - y[j - 1]
                if dy > 1e-12:
                    du_dy[i, j] = (u[i, j + 1] - u[i, j - 1]) / dy
    
    # For thin BL approximation, we use simplified Cartesian gradients:
    # In BL coords: x ~ s (streamwise), y ~ y_normal (wall-normal)
    # So du/dx ~ du/ds and du/dy ~ du/dy_normal
    # This is exact for flat plates, approximate for curved surfaces
    grad_u[:, :, 0, 0] = du_ds    # du/dx
    grad_u[:, :, 0, 1] = du_dy    # du/dy
    # dv/dx and dv/dy remain 0 (thin BL approximation)
    
    return grad_u


def compute_cell_areas(
    x_domain: NDArray[np.float64],
    y_domain: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute cell areas for the domain grid.
    
    Uses a simple rectangular approximation based on grid spacing.
    For point i,j, the area is approximately ds[i] * dy[i,j].
    
    Args:
        x_domain: Physical x-coordinates, shape (M, Ny)
        y_domain: Physical y-coordinates, shape (M, Ny)
    
    Returns:
        areas: Cell areas, shape (M * Ny,) flattened
    """
    M, Ny = x_domain.shape
    areas = np.zeros(M * Ny, dtype=np.float64)
    
    for i in range(M):
        for j in range(Ny):
            # Compute local ds (streamwise spacing)
            if i == 0:
                ds = np.sqrt((x_domain[1, j] - x_domain[0, j])**2 +
                            (y_domain[1, j] - y_domain[0, j])**2)
            elif i == M - 1:
                ds = np.sqrt((x_domain[-1, j] - x_domain[-2, j])**2 +
                            (y_domain[-1, j] - y_domain[-2, j])**2)
            else:
                ds = 0.5 * np.sqrt((x_domain[i+1, j] - x_domain[i-1, j])**2 +
                                   (y_domain[i+1, j] - y_domain[i-1, j])**2)
            
            # Compute local dy (wall-normal spacing)
            if j == 0:
                dy = np.sqrt((x_domain[i, 1] - x_domain[i, 0])**2 +
                            (y_domain[i, 1] - y_domain[i, 0])**2)
            elif j == Ny - 1:
                dy = np.sqrt((x_domain[i, -1] - x_domain[i, -2])**2 +
                            (y_domain[i, -1] - y_domain[i, -2])**2)
            else:
                dy = 0.5 * np.sqrt((x_domain[i, j+1] - x_domain[i, j-1])**2 +
                                   (y_domain[i, j+1] - y_domain[i, j-1])**2)
            
            areas[i * Ny + j] = ds * dy
    
    return areas


def extract_bdim_input_from_bl_field(
    bl_path: "BoundaryLayerPathResult",
    bl_field: "BLFieldData",
    p_surface: Optional[NDArray[np.float64]] = None,
    p_inf: float = 0.0,
) -> BDIMInput:
    """
    Extract BDIM input from viscous BL path and field reconstruction.
    
    Performs the coordinate transformation from (s, y_normal) to (x, y)
    and computes velocity gradients and cell areas.
    
    Args:
        bl_path: BoundaryLayerPathResult from BoundaryLayerRunner
        bl_field: BLFieldData from reconstruct_bl_field()
        p_surface: Surface pressure at each station [Pa], shape (M,).
            If None, assumes p = p_inf everywhere.
        p_inf: Freestream pressure [Pa] for pressure field estimation.
    
    Returns:
        BDIMInput ready for BDIMThermalSolver
    
    Raises:
        ValueError: If field data doesn't match path data
    """
    # Get field dimensions
    M = len(bl_field.s)
    Ny = bl_field.y.shape[1]
    K = M * Ny  # Total domain points
    
    # Get path coordinates at valid stations
    # bl_field.s contains only valid (non-NaN) stations
    # We need to find the corresponding indices in bl_path
    valid_mask = ~np.isnan(bl_path.results[bl_field.profile_name].theta)
    path_x = bl_path.x[valid_mask]
    path_y = bl_path.y[valid_mask]
    
    if len(path_x) != M:
        raise ValueError(
            f"Path length mismatch: field has {M} stations, "
            f"path has {len(path_x)} valid stations"
        )
    
    # Compute outward normals along the path
    normals, tangents = compute_path_normals(path_x, path_y)
    
    # Transform (s, y_normal) grid to physical (x, y)
    x_domain, y_domain = transform_sy_to_xy(
        bl_field.s, bl_field.y, path_x, path_y, normals
    )
    
    # Compute arc-length spacing
    ds = np.diff(bl_field.s)
    
    # Compute velocity gradients
    grad_u = compute_velocity_gradients(bl_field.u, bl_field.y, ds)
    
    # Compute cell areas
    areas = compute_cell_areas(x_domain, y_domain)
    
    # Build domain velocity field
    # u is tangential velocity in BL coords; we approximate:
    # u_x ~ u * cos(theta), u_y ~ u * sin(theta) where theta is local tangent angle
    # For simplicity, use tangent direction from normals
    u_domain = np.zeros((K, 2), dtype=np.float64)
    for i in range(M):
        # Tangent is perpendicular to normal
        tx = tangents[i, 0]
        ty = tangents[i, 1]
        for j in range(Ny):
            idx = i * Ny + j
            u_domain[idx, 0] = bl_field.u[i, j] * tx
            u_domain[idx, 1] = bl_field.u[i, j] * ty
    
    # Flatten domain coordinates
    nodes_domain = np.column_stack([
        x_domain.ravel(),
        y_domain.ravel()
    ])
    
    # Flatten velocity gradients
    grad_u_flat = grad_u.reshape(K, 2, 2)
    
    # Pressure field: assume constant across thin BL (use surface pressure)
    if p_surface is None:
        p_domain = np.full(K, p_inf)
    else:
        # Extend surface pressure across BL thickness
        p_domain = np.zeros(K, dtype=np.float64)
        for i in range(M):
            for j in range(Ny):
                p_domain[i * Ny + j] = p_surface[i]
    
    # Boundary (surface) data
    nodes_b = np.column_stack([path_x, path_y])
    
    # Compute panel lengths
    lengths_b = np.zeros(M, dtype=np.float64)
    lengths_b[:-1] = np.sqrt(np.diff(path_x)**2 + np.diff(path_y)**2)
    lengths_b[-1] = lengths_b[-2]  # Last panel same as previous
    
    # Boundary velocity (surface tangential velocity)
    u_b = np.zeros((M, 2), dtype=np.float64)
    for i in range(M):
        tx = tangents[i, 0]
        ty = tangents[i, 1]
        # Surface velocity is 0 (no-slip), but we use Ue for the "edge" reference
        # Actually at wall, u=0. But BDIM needs the velocity at boundary nodes.
        # For now, set to 0 (no-slip wall condition)
        u_b[i, 0] = 0.0
        u_b[i, 1] = 0.0
    
    return BDIMInput(
        arc_length=bl_field.s,
        nodes_b=nodes_b,
        normals_b=normals,
        lengths_b=lengths_b,
        u_b=u_b,
        x_b=path_x,
        y_b=path_y,
        nodes_domain=nodes_domain,
        areas_domain=areas,
        u_domain=u_domain,
        grad_u_domain=grad_u_flat,
        p_domain=p_domain,
        side=bl_path.side,
        grid_shape=(M, Ny),
        y_normal=bl_field.y,
    )


@dataclass
class BDIMFieldData:
    """
    Domain field data from BDIM thermal solver.
    
    Contains the temperature field in both physical (x, y) and
    boundary layer (s, y_normal) coordinates.
    
    Attributes:
        s: Arc-length stations [m], shape (M,)
        y_normal: Wall-normal distances [m], shape (M, Ny)
        x: Physical x-coordinates [m], shape (M, Ny)
        y: Physical y-coordinates [m], shape (M, Ny)
        T: Temperature field [K], shape (M, Ny)
        side: Surface side identifier
    """
    s: NDArray[np.float64]
    y_normal: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    T: NDArray[np.float64]
    side: str = ""
