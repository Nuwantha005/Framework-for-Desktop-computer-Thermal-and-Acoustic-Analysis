"""
Vectorized assembly of BDIM matrices.

Uses NumPy broadcasting for efficient computation of Green's function
interactions between boundary and domain points.
"""

import numpy as np
from numpy.typing import NDArray


def _compute_distances(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """
    Compute pairwise distances and vectors between point sets.
    
    Args:
        x: Source points, shape (M, 2)
        y: Field points, shape (N, 2)
    
    Returns:
        r: Distance matrix, shape (M, N)
        vec: Vector differences (y - x), shape (M, N, 2)
    """
    # vec[i, j] = y[j] - x[i], shape (M, N, 2)
    vec = y[None, :, :] - x[:, None, :]
    # r[i, j] = ||y[j] - x[i]||, shape (M, N)
    r = np.linalg.norm(vec, axis=2)
    return r, vec


def _temp_fundamental_vectorized(r: NDArray) -> NDArray:
    """
    T*(x, y) = (1 / 2π) * ln(1/r)
    
    Args:
        r: Distance matrix, shape (M, N)
    
    Returns:
        T*: Fundamental solution, shape (M, N)
    """
    # Avoid log(0) by masking small r
    r_safe = np.where(r < 1e-12, 1.0, r)
    T_star = (1.0 / (2.0 * np.pi)) * np.log(1.0 / r_safe)
    # Set singular points to 0
    T_star = np.where(r < 1e-12, 0.0, T_star)
    return T_star


def _temp_derivative_vectorized(r: NDArray, vec: NDArray) -> NDArray:
    """
    T*,i(x, y) = (-1 / 2πr²) * (y_i - x_i)
    
    Args:
        r: Distance matrix, shape (M, N)
        vec: Vector differences, shape (M, N, 2)
    
    Returns:
        grad_T*: Gradient of fundamental solution, shape (M, N, 2)
    """
    r_safe = np.where(r < 1e-12, 1.0, r)
    coeff = -1.0 / (2.0 * np.pi * r_safe**2)
    grad = coeff[:, :, None] * vec
    # Set singular points to 0
    mask = r < 1e-12
    grad[mask, :] = 0.0
    return grad


def assemble_boundary_matrices(
    nodes_b: NDArray, 
    normals_b: NDArray, 
    lengths_b: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Assemble BEM matrices [H] and [G] for boundary equations.
    
    Vectorized implementation for efficiency.
    
    Args:
        nodes_b: Boundary node positions, shape (N, 2)
        normals_b: Boundary outward normals, shape (N, 2)
        lengths_b: Boundary panel lengths, shape (N,)
    
    Returns:
        H: Shape (N, N)
        G: Shape (N, N)
    """
    N = len(nodes_b)
    
    # Compute pairwise distances and vectors
    r, vec = _compute_distances(nodes_b, nodes_b)
    
    # Fundamental solution T*
    T_star = _temp_fundamental_vectorized(r)
    
    # Gradient of T*
    grad_T_star = _temp_derivative_vectorized(r, vec)
    
    # Normal derivative: T*,n = grad_T* · n
    # grad_T_star[i, j, :] · normals_b[j, :] for each (i, j)
    T_star_n = np.sum(grad_T_star * normals_b[None, :, :], axis=2)
    
    # Assemble H and G (off-diagonal)
    H = T_star_n * lengths_b[None, :]
    G = -T_star * lengths_b[None, :]
    
    # Diagonal elements (analytical treatment)
    diag_idx = np.arange(N)
    H[diag_idx, diag_idx] = 0.5  # c(x) = 0.5 for smooth boundary
    # Logarithmic integral for self-influence
    L = lengths_b
    G[diag_idx, diag_idx] = (L / (2.0 * np.pi)) * (1.0 - np.log(L / 2.0))
    
    return H, G


def assemble_domain_matrices(
    nodes_eval: NDArray, 
    nodes_domain: NDArray, 
    areas_domain: NDArray
) -> NDArray:
    """
    Assemble domain mapping matrix [E] for internal thermal effects.
    
    Vectorized implementation.
    
    Args:
        nodes_eval: Evaluation points, shape (M, 2)
        nodes_domain: Domain points, shape (K, 2)
        areas_domain: Domain cell areas, shape (K,)
    
    Returns:
        E: Shape (M, K, 2)
    """
    # Compute pairwise distances and vectors
    r, vec = _compute_distances(nodes_eval, nodes_domain)
    
    # Gradient of T*
    grad_T_star = _temp_derivative_vectorized(r, vec)
    
    # E[i, j, :] = -grad_T*[i, j, :] * areas[j]
    E = -grad_T_star * areas_domain[None, :, None]
    
    # Handle self-influence (diagonal when M == K and same points)
    # Set to 0 for coincident points
    if len(nodes_eval) == len(nodes_domain):
        diag_mask = r < 1e-12
        E[diag_mask, :] = 0.0
    
    return E


def assemble_boundary_domain_coupling(
    nodes_eval: NDArray, 
    nodes_b: NDArray, 
    normals_b: NDArray, 
    lengths_b: NDArray
) -> NDArray:
    """
    Assemble boundary-domain coupling matrix.
    
    Vectorized implementation.
    
    Args:
        nodes_eval: Evaluation points, shape (M, 2)
        nodes_b: Boundary points, shape (N, 2)
        normals_b: Boundary normals, shape (N, 2)
        lengths_b: Boundary panel lengths, shape (N,)
    
    Returns:
        EC: Shape (M, N, 2)
    """
    M = len(nodes_eval)
    N = len(nodes_b)
    
    # Compute pairwise distances
    r, _ = _compute_distances(nodes_eval, nodes_b)
    
    # Fundamental solution T*
    T_star = _temp_fundamental_vectorized(r)
    
    # Handle diagonal self-influence when M == N
    if M == N:
        diag_idx = np.arange(N)
        diag_mask = r[diag_idx, diag_idx] < 1e-12
        # For diagonal: T* = (1/2π) * ln(2/L)
        T_star[diag_idx[diag_mask], diag_idx[diag_mask]] = (
            (1.0 / (2.0 * np.pi)) * np.log(2.0 / lengths_b[diag_mask])
        )
    
    # EC[i, j, :] = T*[i, j] * normals_b[j, :] * lengths_b[j]
    EC = T_star[:, :, None] * normals_b[None, :, :] * lengths_b[None, :, None]
    
    return EC
